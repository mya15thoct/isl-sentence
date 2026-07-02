"""Publication figures for the HARP paper (results only; numbers hard-coded from
paper_facts.md). Produces PDF (vector, for LaTeX) and PNG (300 dpi).

    python scripts/plot_results.py

Colours use the Okabe-Ito colourblind-safe palette. Single-series charts use one
hue (magnitude); the two-series pool chart uses the blue/orange pair.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.8,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
BLUE, ORANGE, INK = "#0072B2", "#E69F00", "#2b2b2b"


def _label(ax, bars, fmt="{:.3f}", dy=1.5):
    for r in bars:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8, color=INK,
                    xytext=(0, dy), textcoords="offset points")


def fig_pool():  # (c) R@10 vs pool size, single vs ensemble  -> defends "low numbers"
    pools = ["Full\n(11,934)", "2,000", "1,000"]
    single = [0.561, 0.735, 0.805]
    ensemble = [0.614, 0.778, 0.842]
    x = np.arange(len(pools)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    b1 = ax.bar(x - w / 2, single, w, label="Single (3-seed mean)", color=BLUE, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, ensemble, w, label="Ensemble (3 seeds)", color=ORANGE, edgecolor="white", linewidth=0.8)
    _label(ax, b1); _label(ax, b2)
    ax.set_ylabel("Recall@10 (test, V2T, Sinkhorn)"); ax.set_xlabel("Candidate pool size")
    ax.set_xticks(x); ax.set_xticklabels(pools); ax.set_ylim(0.45, 0.92)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.xaxis.grid(False); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig("fig_pool.pdf"); fig.savefig("fig_pool.png", dpi=300)


def fig_ablation():  # (b) component ablation, full-pool R@10
    names = ["HARP\n(reference)", "$-$ hand-aware", "$-$ redundancy", "bge $\\to$ MiniLM"]
    r10 = [0.564, 0.518, 0.558, 0.561]
    colors = [BLUE, ORANGE, BLUE, BLUE]  # highlight the big drop
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    bars = ax.bar(range(len(names)), r10, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    _label(ax, bars)
    ax.axhline(0.564, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_ylabel("Recall@10 (test, V2T, Sinkhorn)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.45, 0.60); ax.xaxis.grid(False); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig("fig_ablation.pdf"); fig.savefig("fig_ablation.png", dpi=300)


def fig_ladder():  # (a) input-stream ladder, full-pool R@10
    names = ["Hands\nonly", "+ face", "+ pose", "+ pose\n+ face"]
    r10 = [0.542, 0.553, 0.556, 0.564]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bars = ax.bar(range(len(names)), r10, color=BLUE, width=0.6, edgecolor="white", linewidth=0.8)
    bars[-1].set_color(ORANGE)  # highlight full model
    _label(ax, bars)
    ax.axhline(0.518, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
    ax.annotate("uniform-fusion baseline (0.518)", (0.02, 0.520), fontsize=8, color="0.4")
    ax.set_ylabel("Recall@10 (test, V2T, Sinkhorn)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.50, 0.58); ax.xaxis.grid(False); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig("fig_ladder.pdf"); fig.savefig("fig_ladder.png", dpi=300)


if __name__ == "__main__":
    fig_pool(); fig_ablation(); fig_ladder()
    print("saved fig_pool / fig_ablation / fig_ladder (.pdf + .png)")
