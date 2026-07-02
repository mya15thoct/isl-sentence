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
BLUE, ORANGE, INK = "#0072B2", "#D55E00", "#1a1a1a"      # vivid, colourblind-safe hands
FACE_C, POSE_C, POSE_PT = "#dcdcdc", "#9e9e9e", "#6f6f6f"  # recessive context
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
UPPER = [11, 12, 13, 14, 15, 16, 23, 24]                  # keep upper body only (no legs)
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

    fig, ax = plt.subplots(figsize=(4.4, 5.0))
    ax.scatter(face[:, 0], face[:, 1], s=2.5, color=FACE_C, zorder=1, label="Face (468)")
    for a, b in POSE_EDGES:                                   # upper-body torso/arms only
        ax.plot(*zip(pose[a], pose[b]), color=POSE_C, lw=1.8, zorder=2)
    ax.scatter(pose[UPPER, 0], pose[UPPER, 1], s=18, color=POSE_PT, zorder=3,
               edgecolor="white", linewidth=0.6, label="Pose (upper body)")
    for hand, col, lab in ((lh, BLUE, "Left hand (21)"), (rh, ORANGE, "Right hand (21)")):
        if np.any(hand):
            for a, b in HAND_EDGES:
                ax.plot(*zip(hand[a], hand[b]), color=col, lw=2.6, zorder=4)
            ax.scatter(hand[:, 0], hand[:, 1], s=36, color=col, zorder=5,
                       edgecolor="white", linewidth=0.7, label=lab)

    # normalisation anchor: shoulder midpoint (c) and width (d), neutral ink
    ls, rs = pose[11], pose[12]
    c = (ls + rs) / 2
    ax.plot([ls[0], rs[0]], [ls[1], rs[1]], color=INK, lw=1.5, ls="--", zorder=6)
    ax.scatter(*c, s=75, marker="x", color=INK, zorder=7, linewidths=2.2)
    ax.annotate(r"$(c_x, c_y)$", c, textcoords="offset points", xytext=(7, 6),
                color=INK, fontsize=11)
    ax.annotate(r"width $d$", ((ls[0] + rs[0]) / 2, min(ls[1], rs[1])), textcoords="offset points",
                xytext=(0, -15), ha="center", color=INK, fontsize=11)

    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=8.5,
              bbox_to_anchor=(0.5, -0.04), handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(args.out); fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"saved {args.out} / .png")


if __name__ == "__main__":
    main()
