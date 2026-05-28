"""
Split the iSign manifest into category-specific CSV files.

Input is produced by:
  src/data/build_isign_manifest.py

Example:
  python src/data/split_isign_manifest.py \
      --manifest data/manifests/isign_metadata.csv \
      --output-dir data/subsets
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path("data/manifests/isign_metadata.csv")
DEFAULT_OUTPUT_DIR = Path("data/subsets")

CATEGORY_TO_FILENAME = {
    "word": "isign_words.csv",
    "description": "isign_descriptions.csv",
    "example_sentence": "isign_examples.csv",
    "general_sentence": "isign_general_sentences.csv",
    "other": "isign_other.csv",
}


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_manifest(args.manifest)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("category", "other") or "other"].append(row)

    counts = Counter()
    for category, category_rows in sorted(grouped.items()):
        filename = CATEGORY_TO_FILENAME.get(category, f"isign_{category}.csv")
        output_path = args.output_dir / filename
        write_csv(output_path, fieldnames, category_rows)
        counts[category] = len(category_rows)
        print(f"{category:18s} {len(category_rows):7d} -> {output_path}")

    print(f"Total              {sum(counts.values()):7d}")


if __name__ == "__main__":
    main()
