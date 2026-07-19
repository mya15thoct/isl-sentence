"""Parameter counts for HARP vs every baseline (review Issue 1, defense Q3).

Prints a per-component table (frame encoder / temporal stack / pooling +
projection / total) for each pose-encoder variant used in the paper, plus the
relative gap to HARP. This backs the "parameter-matched uniform fusion" claim:
if the uniform baseline underperforms, it is not because it is smaller.

Pose side only by default (no HuggingFace download); pass --text-model to also
count the (shared) text encoder.

Run:  python scripts/param_count.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.pose_encoder import KeypointConformerEncoder


def component_counts(model: KeypointConformerEncoder) -> dict[str, int]:
    def count(module) -> int:
        return sum(p.numel() for p in module.parameters())

    return {
        "frame_encoder": count(model.frame_encoder),
        "temporal (downsample+conformer)": count(model.downsample) + count(model.layers),
        "pool+projection": count(model.pool) + count(model.projection),
        "total": count(model),
    }


VARIANTS: dict[str, dict] = {
    # name -> KeypointConformerEncoder kwargs (embedding/projection dim set below)
    "HARP (hand-aware, full context)": dict(hand_aware=True, context_parts=("pose", "face")),
    "uniform fusion (no hand-aware)": dict(hand_aware=False),
    "hands-only (no context)": dict(hand_aware=True, context_parts=()),
    "hands+pose": dict(hand_aware=True, context_parts=("pose",)),
    "hands+face": dict(hand_aware=True, context_parts=("face",)),
    "CLIP4Clip-meanP-style (uniform, 0 layers, mean pool)": dict(
        hand_aware=False, num_layers=0, pool_type="mean"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--text-model", help="optional: also count a text encoder (downloads weights)")
    args = parser.parse_args()

    counts: dict[str, dict[str, int]] = {}
    for name, kwargs in VARIANTS.items():
        kwargs = dict(kwargs)
        kwargs.setdefault("num_layers", args.layers)
        model = KeypointConformerEncoder(
            model_dim=args.model_dim,
            projection_dim=args.embedding_dim,
            num_heads=args.heads,
            **kwargs,
        )
        counts[name] = component_counts(model)

    reference = counts["HARP (hand-aware, full context)"]["total"]
    header = "| variant | frame encoder | temporal | pool+proj | total | vs HARP |"
    print(header)
    print("|" + "---|" * 6)
    for name, c in counts.items():
        gap = 100.0 * (c["total"] - reference) / reference
        print(
            f"| {name} | {c['frame_encoder']:,} | {c['temporal (downsample+conformer)']:,} "
            f"| {c['pool+projection']:,} | {c['total']:,} | {gap:+.1f}% |"
        )

    print(
        "\nNote: hand-aware context ablations (hands-only / hands+pose / hands+face) "
        "instantiate all context branches but only USE the listed ones, so their "
        "counts equal HARP's. The headline claim is HARP vs uniform fusion."
    )

    if args.text_model:
        from src.models.text_encoder import TextEncoder

        text = TextEncoder(model_name=args.text_model, output_dim=args.embedding_dim)
        total = sum(p.numel() for p in text.parameters())
        print(f"\ntext encoder ({args.text_model}): {total:,} params (shared by all variants)")


if __name__ == "__main__":
    main()
