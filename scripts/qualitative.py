"""Qualitative retrieval figure: for a few test queries, render the pose skeleton
(hands emphasised, body faint --- matching the hand-aware design) and list the
top-5 retrieved captions with the ground truth highlighted.

    python scripts/qualitative.py \
        --checkpoint /mnt/recover/ngan/ISL-Sequences/checkpoints/abl_ema50/checkpoint_ema.pt \
        --manifest   /mnt/recover/ngan/ISL-Sequences/manifests/test_recover.csv \
        --examples 4 --out qualitative.pdf

Video was pose-only from the start, so we visualise the MediaPipe skeleton, not RGB.
"""
from __future__ import annotations
import argparse, sys, textwrap
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.retrieval.dataset import RetrievalDataset, collate_retrieval
from src.retrieval.evaluate_rerank import build_model_from_checkpoint, encode_manifest
from torch.utils.data import DataLoader

# keypoint layout: pose[0:132]=33x4(x,y,z,vis); face[132:1536]; hands[1536:1662]=2x21x3
POSE, FACE = 132, 1404
HAND = 63
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#1a9850", "#b0b0b0"
# minimal connections for a readable stick figure
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
              (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def _pts(block: np.ndarray, n: int, stride: int) -> np.ndarray:
    x = block[0 : n * stride : stride]
    y = block[1 : n * stride : stride]
    return np.stack([x, -y], axis=1)  # flip y: image coords -> upright


def draw_skeleton(ax, frame: np.ndarray):
    pose = _pts(frame[:POSE], 33, 4)
    lh = _pts(frame[POSE + FACE : POSE + FACE + HAND], 21, 3)
    rh = _pts(frame[POSE + FACE + HAND :], 21, 3)
    for a, b in POSE_EDGES:  # faint body
        ax.plot(*zip(pose[a], pose[b]), color=GREY, lw=1.4, zorder=1)
    ax.scatter(pose[:, 0], pose[:, 1], s=6, color=GREY, zorder=2)
    for hand, col in ((lh, BLUE), (rh, ORANGE)):  # emphasised hands
        if np.any(hand):
            for a, b in HAND_EDGES:
                ax.plot(*zip(hand[a], hand[b]), color=col, lw=1.6, zorder=3)
            ax.scatter(hand[:, 0], hand[:, 1], s=10, color=col, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("qualitative.pdf"))
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ds = RetrievalDataset(manifest=args.manifest, text_column="canonical_text",
                          max_frames=512, sample_mode="uniform")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, collate_fn=collate_retrieval)
    model = build_model_from_checkpoint(args.checkpoint, device)
    video, text, _ = encode_manifest(model, loader, device)          # (N,E) normalised
    sim = (video @ text.T)                                            # V2T

    rng = np.random.default_rng(args.seed)
    queries = rng.choice(len(ds), size=args.examples, replace=False)
    texts = [ds.rows[i]["canonical_text"] for i in range(len(ds))]

    fig, axes = plt.subplots(args.examples, 2, figsize=(9, 2.4 * args.examples),
                             gridspec_kw={"width_ratios": [1, 2.4]})
    if args.examples == 1:
        axes = axes[None, :]
    for row, qi in enumerate(queries):
        kp = np.nan_to_num(np.load(ds.rows[qi]["keypoint_path"])).astype(np.float32)
        draw_skeleton(axes[row, 0], kp[len(kp) // 2])                 # middle frame
        top = torch.topk(sim[qi], args.topk).indices.tolist()
        ax = axes[row, 1]; ax.axis("off")
        lines = []
        for rank, ci in enumerate(top, 1):
            ok = ci == qi
            mark = "✓" if ok else " "
            cap = textwrap.shorten(texts[ci], width=60, placeholder=" ...")
            lines.append((f"{rank}. [{mark}] {cap}", GREEN if ok else "#333333", ok))
        for j, (t, col, ok) in enumerate(lines):
            ax.text(0.0, 0.9 - j * 0.19, t, color=col, fontsize=10,
                    fontweight="bold" if ok else "normal", transform=ax.transAxes, va="top")
        gt = textwrap.shorten(texts[qi], width=55, placeholder=" ...")
        axes[row, 0].set_title(f"Query: “{gt}”", fontsize=9, color="#333333")
    fig.suptitle("Top-5 retrieved captions per query (✓ = ground truth)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out); fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"saved {args.out} / .png")


if __name__ == "__main__":
    main()
