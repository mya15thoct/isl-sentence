"""Re-split iSign manifests by video_id (the iSign-recommended protocol).

iSign recommends keeping all segments of a ``video_id`` in the same split, with
an 80/10/10 train/val/test ratio. Our original split was per-segment (uid) and
leaks ~82% of videos across splits. This script pools one or more existing
manifests, groups rows by video_id, and assigns whole video groups to
train/val/test so the row counts land near the target ratio with NO video
spanning two splits.

Usage:
    python -m src.data.resplit_by_video \
        --inputs train.csv val.csv test.csv \
        --out-dir .../manifests --out-prefix isign_retrieval_videosplit \
        --ratios 0.8 0.1 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.data.check_split_leakage import video_id


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True,
                    help="manifests to pool (e.g. the old train/val/test)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-prefix", default="isign_retrieval_videosplit")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Pool all rows, keep the union of columns (+ a 'split' column).
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in args.inputs:
        part, cols = read_csv(path)
        for c in cols:
            if c not in fieldnames:
                fieldnames.append(c)
        rows.extend(part)
    if "split" not in fieldnames:
        fieldnames.append("split")

    # Group by video_id.
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[video_id(r)].append(r)
    video_ids = list(groups)
    random.Random(args.seed).shuffle(video_ids)

    total = len(rows)
    train_cap = args.ratios[0] * total
    val_cap = (args.ratios[0] + args.ratios[1]) * total

    # Greedily fill train, then val, then test by accumulating whole video groups.
    assignment: dict[str, str] = {}
    running = 0
    for vid in video_ids:
        if running < train_cap:
            split = "train"
        elif running < val_cap:
            split = "val"
        else:
            split = "test"
        assignment[vid] = split
        running += len(groups[vid])

    buckets: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for vid, grp in groups.items():
        split = assignment[vid]
        for r in grp:
            r = dict(r)
            r["split"] = split
            buckets[split].append(r)

    for split, bucket in buckets.items():
        path = args.out_dir / f"{args.out_prefix}_{split}.csv"
        write_csv(path, bucket, fieldnames)
        n_vids = sum(1 for v, s in assignment.items() if s == split)
        print(f"{split:6s}: {len(bucket):>7d} rows ({len(bucket)/total*100:4.1f}%) | "
              f"{n_vids:>6d} videos | {path.name}")
    print(f"\ntotal: {total} rows, {len(video_ids)} videos. Verify with check_split_leakage.")


if __name__ == "__main__":
    main()
