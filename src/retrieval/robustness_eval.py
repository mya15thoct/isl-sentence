"""Inference-time robustness evaluation (review Issue 5, fix option 3).

Measures how retrieval degrades under controlled input corruptions applied at
TEST time only (no retraining): the questions "what happens when hand detection
is intermittent / the frame rate drops / keypoints are noisy" answered with the
released checkpoint itself.

Perturbations (text side untouched; encoded once):
  * clean                 - reference numbers (must match the main table);
  * hand_drop p           - zero BOTH hands in a random fraction p of frames,
                            simulating intermittent hand detection;
  * frame_sub k           - keep every k-th frame, simulating a lower frame rate;
  * noise s               - add N(0, s) to all keypoint coordinates, simulating
                            detector jitter (inputs are shoulder-normalized).

Reports exact V2T/T2V R@1/R@5/R@10 + median under cosine AND Sinkhorn for every
configuration. GPU recommended (one full TEST encode per configuration).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.analysis_common import per_query_ranks, recall_at, rerank
from src.retrieval.dataset import RetrievalDataset, collate_retrieval
from src.retrieval.evaluate_rerank import build_model_from_checkpoint, checkpoint_input_flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hand-drop", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    parser.add_argument("--frame-sub", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--noise", type=float, nargs="*", default=[0.02])
    parser.add_argument("--sinkhorn-temp", type=float, default=20.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, help="Markdown output")
    parser.add_argument("--out", type=Path, help="JSON output")
    return parser.parse_args()


HANDS_START = 1536  # pose 132 + face 1404; hands = [1536:1662]


def perturb(
    keypoints: torch.Tensor,
    lengths: torch.Tensor,
    kind: str,
    value: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one corruption to a padded (B, T, 1662) batch."""
    if kind == "clean":
        return keypoints, lengths
    if kind == "hand_drop":
        drop = torch.rand(keypoints.shape[:2], generator=generator) < value  # (B, T)
        keypoints = keypoints.clone()
        keypoints[..., HANDS_START:] = keypoints[..., HANDS_START:].masked_fill(
            drop.unsqueeze(-1), 0.0
        )
        return keypoints, lengths
    if kind == "frame_sub":
        k = int(value)
        return keypoints[:, ::k], torch.clamp((lengths + k - 1) // k, min=1)
    if kind == "noise":
        return keypoints + value * torch.randn(keypoints.shape, generator=generator), lengths
    raise ValueError(f"unknown perturbation {kind!r}")


@torch.no_grad()
def encode_videos(model, loader, device, kind: str, value: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)  # CPU-side, reproducible
    chunks = []
    for batch in loader:
        keypoints, lengths = perturb(batch["keypoints"], batch["lengths"], kind, value, generator)
        video, _, _ = model.encode_pose(
            keypoints.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        )
        chunks.append(video.float().cpu())
    return F.normalize(torch.cat(chunks), dim=-1, eps=1e-6)


@torch.no_grad()
def encode_texts(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    chunks, groups = [], []
    for batch in loader:
        tokens = model.text_encoder.tokenize(batch["texts"])
        tokens = {key: value.to(device) for key, value in tokens.items()}
        chunks.append(model.encode_text(tokens["input_ids"], tokens["attention_mask"]).float().cpu())
        groups.append(batch["group_ids"])
    return F.normalize(torch.cat(chunks), dim=-1, eps=1e-6), torch.cat(groups)


def metrics_row(sim, group_ids, sinkhorn_temp, sinkhorn_iters) -> dict[str, float]:
    out: dict[str, float] = {}
    for method in ("cosine", "sinkhorn"):
        for direction, scores in (("v2t", sim), ("t2v", sim.T)):
            ranked = rerank(scores, method, sinkhorn_temp=sinkhorn_temp, sinkhorn_iters=sinkhorn_iters)
            ranks = per_query_ranks(ranked, group_ids)["exact"]
            for k in (1, 5, 10):
                out[f"{method}_{direction}_r{k}"] = round(recall_at(ranks, k), 4)
            out[f"{method}_{direction}_med"] = float(ranks.float().median())
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    motion, motion_hands, face_keep, input_parts = checkpoint_input_flags(args.checkpoint)
    if motion or motion_hands:
        raise SystemExit("robustness_eval supports position-only checkpoints (hand slice differs with motion features)")
    dataset = RetrievalDataset(
        manifest=args.manifest,
        text_column=args.text_column,
        keypoint_column=args.keypoint_column,
        max_frames=args.max_frames,
        sample_mode="uniform",
        limit=args.limit,
        face_keep=face_keep,
        input_parts=input_parts,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_retrieval,
                        pin_memory=device.type == "cuda")
    model = build_model_from_checkpoint(args.checkpoint, device)
    print(f"rows: {len(dataset)}  checkpoint: {args.checkpoint}")

    text, group_ids = encode_texts(model, loader, device)

    configs: list[tuple[str, float]] = [("clean", 0.0)]
    configs += [("hand_drop", p) for p in args.hand_drop]
    configs += [("frame_sub", float(k)) for k in args.frame_sub]
    configs += [("noise", s) for s in args.noise]

    rows = []
    for kind, value in configs:
        label = kind if kind == "clean" else f"{kind}={value:g}"
        print(f"encoding: {label} ...", flush=True)
        video = encode_videos(model, loader, device, kind, value, args.seed)
        rows.append({"config": label, **metrics_row(video @ text.T, group_ids,
                                                    args.sinkhorn_temp, args.sinkhorn_iters)})

    header = ("| config | cos v2t R@10 | sink v2t R@1 | R@5 | R@10 | med | sink t2v R@10 | ΔR@10 vs clean |")
    lines = ["# Robustness to input corruption (inference-time)", "", header,
             "|" + "---|" * 8]
    clean_r10 = rows[0]["sinkhorn_v2t_r10"]
    for r in rows:
        delta = r["sinkhorn_v2t_r10"] - clean_r10
        lines.append(
            f"| {r['config']} | {r['cosine_v2t_r10']:.3f} | {r['sinkhorn_v2t_r1']:.3f} "
            f"| {r['sinkhorn_v2t_r5']:.3f} | {r['sinkhorn_v2t_r10']:.3f} "
            f"| {r['sinkhorn_v2t_med']:.0f} | {r['sinkhorn_t2v_r10']:.3f} | {delta:+.3f} |"
        )
    report = "\n".join(lines)
    print("\n" + report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"\nreport saved: {args.report}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"summary saved: {args.out}")


if __name__ == "__main__":
    main()
