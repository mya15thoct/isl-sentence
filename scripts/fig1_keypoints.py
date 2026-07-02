"""Figure 1 --- one image. Renders a real MediaPipe Holistic skeleton from one
keypoint file: dense face (faint), body pose (grey), hands (emphasised), and the
shoulder-midpoint / shoulder-width normalisation anchor. No GPU needed.

    python scripts/fig1_keypoints.py \
        --keypoint /mnt/recover/ngan/ISL-Sequences/isign_keypoints/general_sentence/<uid>.npy \
        --out fig1_keypoints.pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POSE, FACE, HAND = 132, 1404, 63
BLUE, ORANGE, GREY, FAINT, ANCHOR = "#0072B2", "#E69F00", "#8a8a8a", "#d9d9d9", "#cc3311"
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
              (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def pts(block, n, stride):
    return np.stack([block[0:n * stride:stride], -block[1:n * stride:stride]], axis=1)  # flip y upright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keypoint", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=-1, help="frame index; -1 = middle")
    ap.add_argument("--out", type=Path, default=Path("fig1_keypoints.pdf"))
    args = ap.parse_args()

    kp = np.nan_to_num(np.load(args.keypoint)).astype(np.float32)
    f = kp[len(kp) // 2 if args.frame < 0 else args.frame]
    pose = pts(f[:POSE], 33, 4)
    face = pts(f[POSE:POSE + FACE], 468, 3)
    lh = pts(f[POSE + FACE:POSE + FACE + HAND], 21, 3)
    rh = pts(f[POSE + FACE + HAND:], 21, 3)

    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    ax.scatter(face[:, 0], face[:, 1], s=3, color=FAINT, zorder=1, label="Face (468)")
    for a, b in POSE_EDGES:
        ax.plot(*zip(pose[a], pose[b]), color=GREY, lw=1.6, zorder=2)
    ax.scatter(pose[:, 0], pose[:, 1], s=10, color=GREY, zorder=3, label="Pose (33)")
    for hand, col, lab in ((lh, BLUE, "Left hand (21)"), (rh, ORANGE, "Right hand (21)")):
        if np.any(hand):
            for a, b in HAND_EDGES:
                ax.plot(*zip(hand[a], hand[b]), color=col, lw=2.0, zorder=4)
            ax.scatter(hand[:, 0], hand[:, 1], s=16, color=col, zorder=5, label=lab)

    # normalisation anchor: shoulder midpoint (c) and width (d)
    ls, rs = pose[11], pose[12]
    c = (ls + rs) / 2
    ax.plot([ls[0], rs[0]], [ls[1], rs[1]], color=ANCHOR, lw=1.6, ls="--", zorder=6)
    ax.scatter(*c, s=60, marker="x", color=ANCHOR, zorder=7, linewidths=2)
    ax.annotate(r"$(c_x, c_y)$", c, textcoords="offset points", xytext=(6, 6),
                color=ANCHOR, fontsize=11)
    ax.annotate(r"width $d$", ((ls[0] + rs[0]) / 2, ls[1]), textcoords="offset points",
                xytext=(0, -16), ha="center", color=ANCHOR, fontsize=11)

    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=8,
              bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(args.out); fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"saved {args.out} / .png")


if __name__ == "__main__":
    main()
