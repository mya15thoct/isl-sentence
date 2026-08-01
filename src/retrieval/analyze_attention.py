"""Interpretability: what does the hand-aware cross-attention learn? (paper §6.x)

The hand-aware encoder queries pose/face *context* from the hand backbone via a
per-frame single-head attention. This script turns that attention on
(``capture_attn``), runs the trained model over a manifest, and reports how much
the model leans on the pose vs face context — overall, and stratified by caption
type, caption length, and (if available) the clip's hand-detection ratio.

The scientific question: is the context attention behaving as the linguistic
motivation predicts — i.e. does the model pull more on the FACE stream for
caption types that carry non-manual markers, and lean on POSE context when the
hands are less reliably detected? A positive signal turns "hand-aware" from a
generic weighted fusion into a mechanism with an interpretable, sign-linguistic
role. Runs on a trained hand-aware checkpoint with pose+face context; CPU-ok.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.dataset import RetrievalDataset, collate_retrieval, read_csv
from src.retrieval.evaluate_rerank import build_model_from_checkpoint, checkpoint_input_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="subsample rows for a quick run")
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--hand-ratio-column", default="both_hand_frame_ratio")
    parser.add_argument("--length-bins", type=int, nargs="+", default=[4, 8, 15])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def bin_label(v: int, edges: list[int]) -> str:
    lo = None
    for e in edges:
        if v <= e:
            return f"{(lo + 1) if lo is not None else 0}-{e}"
        lo = e
    return f">{edges[-1]}"


def group_means(labels: list[str], values: list[float], order=None) -> list[tuple[str, int, float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for lab, v in zip(labels, values):
        buckets[lab].append(v)
    keys = order or sorted(buckets)
    return [(k, len(buckets[k]), sum(buckets[k]) / len(buckets[k])) for k in keys if k in buckets]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    motion, motion_hands, face_keep, input_parts = checkpoint_input_flags(args.checkpoint)
    model = build_model_from_checkpoint(args.checkpoint, device)
    fenc = model.pose_encoder.frame_encoder
    if not hasattr(fenc, "capture_attn"):
        raise SystemExit("checkpoint is not hand-aware — no context attention to analyse")
    parts = list(fenc.context_parts)
    if len(parts) < 2:
        raise SystemExit(f"need pose+face context to compare attention; got {parts}")
    fenc.capture_attn = True
    print(f"context parts (attention keys): {parts}")

    dataset = RetrievalDataset(
        manifest=args.manifest, text_column=args.text_column, keypoint_column=args.keypoint_column,
        max_frames=args.max_frames, sample_mode="uniform", limit=args.limit,
        motion_features=motion, motion_hands=motion_hands, face_keep=face_keep, input_parts=input_parts,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_retrieval,
                        pin_memory=device.type == "cuda")

    # Extra per-row metadata joined by row order (dataset preserves manifest order).
    rows = read_csv(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    per_part_attn: dict[str, list[float]] = {p: [] for p in parts}
    captions: list[str] = []

    with torch.no_grad():
        for batch in loader:
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            model.encode_pose(keypoints, lengths)
            attn = fenc._last_attn  # (B, T, N) at raw frame resolution
            # mask to valid frames, mean over time -> (B, N)
            T = attn.size(1)
            valid = (torch.arange(T, device=attn.device)[None, :] < lengths[:, None].to(attn.device)).float()
            mean_attn = (attn * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            mean_attn = mean_attn.cpu()
            for i, p in enumerate(parts):
                per_part_attn[p].extend(mean_attn[:, i].tolist())
            captions.extend(batch["texts"])

    n = len(captions)
    face_attn = per_part_attn.get("face", [0.0] * n)
    pose_attn = per_part_attn.get("pose", [0.0] * n)

    summary: dict = {"n": n, "context_parts": parts,
                     "overall_mean_attention": {p: sum(v) / len(v) for p, v in per_part_attn.items()}}
    sec: list[str] = [f"# Context-attention analysis — {args.checkpoint.name}",
                      f"clips: {n}  |  attention keys: {parts}\n",
                      "## Overall mean attention (query = hands)"]
    for p, m in summary["overall_mean_attention"].items():
        sec.append(f"- **{p}**: {m:.3f}")
    sec.append("")

    # by caption length
    word_counts = [len(c.split()) for c in captions]
    len_labels = [bin_label(w, args.length_bins) for w in word_counts]
    order = sorted(set(len_labels), key=lambda l: (l.startswith(">"), l))
    face_by_len = group_means(len_labels, face_attn, order)
    sec.append("## Face-context attention by caption length (words)")
    sec.append("| bucket | n | mean face-attn |\n|---|---|---|")
    for k, c, m in face_by_len:
        sec.append(f"| {k} | {c} | {m:.3f} |")
    summary["face_attention_by_length"] = [{"bucket": k, "n": c, "face_attn": m} for k, c, m in face_by_len]
    sec.append("")

    # by category (if column present)
    if rows and args.category_column in rows[0]:
        cats = [r.get(args.category_column, "").strip() or "(none)" for r in rows][:n]
        face_by_cat = group_means(cats, face_attn)
        sec.append(f"## Face-context attention by `{args.category_column}`")
        sec.append("| category | n | mean face-attn |\n|---|---|---|")
        for k, c, m in face_by_cat:
            sec.append(f"| {k} | {c} | {m:.3f} |")
        summary["face_attention_by_category"] = [{"category": k, "n": c, "face_attn": m} for k, c, m in face_by_cat]
        sec.append("")

    # correlation with hand-detection ratio (does pose context rise when hands are scarce?)
    if rows and args.hand_ratio_column in rows[0]:
        try:
            ratios = [float(r.get(args.hand_ratio_column) or "nan") for r in rows][:n]
        except ValueError:
            ratios = []
        pairs = [(r, p) for r, p in zip(ratios, pose_attn) if r == r]  # drop nan
        if len(pairs) > 10:
            import statistics
            rs, ps = zip(*pairs)
            # Pearson correlation
            mr, mp = statistics.mean(rs), statistics.mean(ps)
            cov = sum((a - mr) * (b - mp) for a, b in pairs)
            dr = sum((a - mr) ** 2 for a in rs) ** 0.5
            dp = sum((b - mp) ** 2 for b in ps) ** 0.5
            corr = cov / (dr * dp) if dr > 0 and dp > 0 else float("nan")
            sec.append(f"## Pose-context attention vs hand-detection ratio (`{args.hand_ratio_column}`)")
            sec.append(f"Pearson r = **{corr:+.3f}** over {len(pairs)} clips "
                       f"(negative ⇒ the model leans on pose context MORE when hands are less detected).")
            summary["pose_attn_vs_hand_ratio_pearson"] = corr
            sec.append("")

    report = "\n".join(sec)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"\nreport saved: {args.report}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary saved: {args.out}")


if __name__ == "__main__":
    main()
