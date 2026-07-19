"""Caption-redundancy statistics per split (review Issue 3).

Quantifies the duplicate-caption phenomenon that motivates the
redundancy-aware objective/metric, so the paper can report it as an
evaluation-correctness safeguard with hard numbers:

  * rows, distinct caption groups, % rows whose caption recurs;
  * group-size distribution (max / p50 / p90 / p99) and the top groups;
  * expected number of extra in-batch positives per row at the training
    batch size (how often redundancy grouping actually fires);
  * cross-split caption overlap (test captions also seen in train - textual
    overlap only; video-level leakage is separately zero by construction).

Grouping uses the exact normalization used in training/eval
(``normalize_for_grouping``), so these numbers describe the real protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.dataset import normalize_for_grouping, read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", help="one per manifest (default: file stems)")
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--batch-size", type=int, default=128, help="training batch size for the in-batch-positive estimate")
    parser.add_argument("--top-groups", type=int, default=10)
    parser.add_argument("--out", type=Path, help="optional JSON output")
    return parser.parse_args()


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return float(sorted_values[idx])


def split_stats(rows: list[dict], text_column: str, batch_size: int, top_n: int) -> dict:
    counts: Counter[str] = Counter()
    for row in rows:
        text = row.get(text_column, "").strip()
        if text:
            counts[normalize_for_grouping(text)] += 1

    n = sum(counts.values())
    sizes = sorted(counts.values())
    dup_rows = sum(size for size in sizes if size >= 2)

    # For a row in a group of size s, a random batch of B holds on average
    # (s-1)(B-1)/(N-1) other members of its group. Averaging over rows gives the
    # expected number of extra in-batch positives redundancy grouping marks.
    expected_positives = (
        sum(size * (size - 1) for size in sizes) / n * (batch_size - 1) / max(1, n - 1)
        if n > 1 else 0.0
    )

    return {
        "rows": n,
        "groups": len(counts),
        "duplicate_rows": dup_rows,
        "duplicate_row_fraction": round(dup_rows / max(1, n), 4),
        "group_size_max": sizes[-1] if sizes else 0,
        "group_size_p50": percentile(sizes, 0.50),
        "group_size_p90": percentile(sizes, 0.90),
        "group_size_p99": percentile(sizes, 0.99),
        "expected_extra_inbatch_positives": round(expected_positives, 4),
        "top_groups": [
            {"caption": caption, "count": count} for caption, count in counts.most_common(top_n)
        ],
        "_keys": set(counts),  # stripped before JSON; used for overlap
    }


def main() -> None:
    args = parse_args()
    labels = args.labels or [m.stem for m in args.manifests]
    if len(labels) != len(args.manifests):
        raise SystemExit("--labels must match --manifests")

    stats: dict[str, dict] = {}
    for label, manifest in zip(labels, args.manifests):
        stats[label] = split_stats(read_csv(manifest), args.text_column, args.batch_size, args.top_groups)

    print("| split | rows | caption groups | rows w/ duplicated caption | max group | p99 | E[extra in-batch positives] |")
    print("|---|---|---|---|---|---|---|")
    for label, s in stats.items():
        print(
            f"| {label} | {s['rows']} | {s['groups']} | {s['duplicate_rows']} "
            f"({100 * s['duplicate_row_fraction']:.1f}%) | {s['group_size_max']} "
            f"| {s['group_size_p99']:.0f} | {s['expected_extra_inbatch_positives']:.3f} |"
        )

    # Cross-split caption overlap (textual only; disclose alongside the
    # video-level zero-leakage guarantee).
    overlap: dict[str, dict] = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = stats[a]["_keys"] & stats[b]["_keys"]
            overlap[f"{a}&{b}"] = {
                "shared_groups": len(shared),
                f"share_of_{b}_groups": round(len(shared) / max(1, stats[b]["groups"]), 4),
            }
    print("\nCross-split caption (group) overlap - textual, not video leakage:")
    for pair, o in overlap.items():
        print(f"  {pair}: {o['shared_groups']} shared groups "
              f"({100 * list(o.values())[1]:.1f}% of the smaller side)")

    print("\nTop duplicated captions (test-relevant hubs):")
    for label, s in stats.items():
        top = ", ".join(f"\"{g['caption'][:40]}\"x{g['count']}" for g in s["top_groups"][:3])
        print(f"  {label}: {top}")

    if args.out:
        for s in stats.values():
            s.pop("_keys")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"splits": stats, "overlap": overlap}, indent=2), encoding="utf-8")
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
