"""Create fixed train/val/test CSV splits for SignPose2Text experiments."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


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


def group_value(row: dict[str, str], group_column: str, row_index: int) -> str:
    return (
        row.get(group_column, "").strip()
        or row.get("uid", "").strip()
        or row.get("keypoint_path", "").strip()
        or f"row-{row_index}"
    )


def grouped_rows(
    rows: list[dict[str, str]],
    group_column: str,
) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row_index, row in enumerate(rows):
        groups.setdefault(group_value(row, group_column, row_index), []).append(row)
    return list(groups.values())


def split_groups(
    groups: list[list[dict[str, str]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    total_rows = sum(len(group) for group in shuffled)
    train_target = round(total_rows * train_ratio)
    val_target = round(total_rows * val_ratio)

    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for group in shuffled:
        if len(train) < train_target:
            train.extend(group)
        elif len(val) < val_target:
            val.extend(group)
        else:
            test.extend(group)

    return train, val, test


def mark_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        next_row = dict(row)
        next_row["split"] = split
        output.append(next_row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--group-column", default="canonical_text")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise SystemExit("train/val/test ratios must all be positive.")

    rows, fieldnames = read_csv(args.manifest)
    if not rows:
        raise SystemExit("No rows found.")

    groups = grouped_rows(rows, args.group_column)
    train, val, test = split_groups(groups, args.train_ratio, args.val_ratio, args.seed)
    output_fields = list(fieldnames)
    if "split" not in output_fields:
        output_fields.append("split")

    prefix = args.prefix or args.manifest.stem
    train_path = args.output_dir / f"{prefix}_train.csv"
    val_path = args.output_dir / f"{prefix}_val.csv"
    test_path = args.output_dir / f"{prefix}_test.csv"

    write_csv(train_path, mark_split(train, "train"), output_fields)
    write_csv(val_path, mark_split(val, "val"), output_fields)
    write_csv(test_path, mark_split(test, "test"), output_fields)

    print(f"input rows  : {len(rows)}")
    print(f"groups      : {len(groups)} by {args.group_column}")
    print(f"train rows  : {len(train)} -> {train_path}")
    print(f"val rows    : {len(val)} -> {val_path}")
    print(f"test rows   : {len(test)} -> {test_path}")
    print(f"test ratio  : {test_ratio:.4f}")


if __name__ == "__main__":
    main()
