"""
Build a training-ready manifest from extracted keypoint files.

This script is useful while extraction is still running: it scans the metadata
CSV, checks which expected .npy files already exist, and writes rows that can be
fed into text embedding or contrastive training scripts.

Example:
  python src/data/build_keypoint_manifest.py \
      --metadata /mnt/ngan/ISL-Sequences/manifests/isign_metadata.csv \
      --keypoint-root /mnt/ngan/ISL-Sequences/isign_keypoints \
      --category general_sentence \
      --check-files \
      --only-usable \
      --output /mnt/ngan/ISL-Sequences/manifests/isign_general_existing_clean.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


KEYPOINT_DIM = 1662
USABLE_STATUSES = {"ok", "skipped_exists"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def keypoint_path_for(row: dict[str, str], keypoint_root: Path) -> Path:
    category = row.get("category", "").strip() or "unknown"
    uid = row.get("uid", "").strip()
    return keypoint_root / category / f"{uid}.npy"


def load_keypoint_info(path: Path) -> tuple[str, str, str, str]:
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception as exc:
        return "", "", "read_error", str(exc)

    shape = tuple(int(x) for x in arr.shape)
    shape_text = "x".join(str(x) for x in shape)
    if len(shape) != 2 or shape[1] != KEYPOINT_DIM:
        return str(shape[0]) if shape else "", shape_text, "bad_shape", (
            f"expected Nx{KEYPOINT_DIM}, got {shape_text}"
        )

    return str(shape[0]), shape_text, "ok", ""


def augment_rows(
    rows: list[dict[str, str]],
    keypoint_root: Path,
    categories: set[str],
    check_files: bool,
    only_usable: bool,
    min_frames: int,
    start: int,
    limit: int | None,
) -> list[dict[str, str]]:
    if categories:
        rows = [row for row in rows if row.get("category", "").strip() in categories]

    if start:
        rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]

    output_rows: list[dict[str, str]] = []

    for row in rows:
        out_path = keypoint_path_for(row, keypoint_root)
        next_row = dict(row)
        next_row["keypoint_path"] = str(out_path)
        next_row["keypoint_exists"] = "1" if out_path.is_file() else "0"
        next_row["num_frames"] = ""
        next_row["keypoint_shape"] = ""
        next_row["extract_status"] = ""
        next_row["extract_error"] = ""

        if not out_path.is_file():
            next_row["extract_status"] = "missing_keypoint"
            next_row["extract_error"] = f"keypoint file not found: {out_path}"
        elif check_files:
            num_frames, shape_text, status, error = load_keypoint_info(out_path)
            next_row["num_frames"] = num_frames
            next_row["keypoint_shape"] = shape_text
            next_row["extract_status"] = status
            next_row["extract_error"] = error
        else:
            next_row["extract_status"] = "skipped_exists"

        if next_row["extract_status"] in USABLE_STATUSES:
            try:
                if int(next_row.get("num_frames") or min_frames) < min_frames:
                    next_row["extract_status"] = "too_short"
                    next_row["extract_error"] = (
                        f"num_frames < min_frames ({min_frames})"
                    )
            except ValueError:
                pass

        if only_usable and next_row["extract_status"] not in USABLE_STATUSES:
            continue

        output_rows.append(next_row)

    return output_rows


def ensure_fields(fieldnames: list[str]) -> list[str]:
    fields = list(fieldnames)
    for field in (
        "keypoint_path",
        "keypoint_exists",
        "num_frames",
        "keypoint_shape",
        "extract_status",
        "extract_error",
    ):
        if field not in fields:
            fields.append(field)
    return fields


def print_summary(rows: list[dict[str, str]], output: Path) -> None:
    status_counts = Counter(row.get("extract_status", "") for row in rows)
    category_counts = Counter(row.get("category", "") for row in rows)
    usable = sum(status_counts[status] for status in USABLE_STATUSES)

    print(f"Rows written : {len(rows)}")
    print(f"Usable rows  : {usable}")
    print("By status:")
    for status, count in status_counts.most_common():
        print(f"  {status or '<blank>':18s}: {count}")
    print("By category:")
    for category, count in category_counts.most_common():
        print(f"  {category or '<blank>':18s}: {count}")
    print(f"Manifest     : {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Optional category filter. May be passed multiple times.",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Open each .npy and verify shape/num_frames.",
    )
    parser.add_argument(
        "--only-usable",
        action="store_true",
        help="Write only rows whose keypoints are readable and usable.",
    )
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_csv(args.metadata)
    output_rows = augment_rows(
        rows=rows,
        keypoint_root=args.keypoint_root,
        categories=set(args.category),
        check_files=args.check_files,
        only_usable=args.only_usable,
        min_frames=args.min_frames,
        start=args.start,
        limit=args.limit,
    )

    fields = ensure_fields(fieldnames)
    write_csv(args.output, fields, output_rows)
    print_summary(output_rows, args.output)


if __name__ == "__main__":
    main()
