"""
Build a clean manifest for the iSign video dataset.

This is the first step after resetting src/: before extracting keypoints or
training models, we need one reliable table that maps each CSV uid to:
  - text label
  - video path
  - dataset role/category
  - word-presence / word-description relation when available

Example:
  python src/data/build_isign_manifest.py \
      --csv-dir .csv \
      --video-root /mnt/ngan/isign_videos \
      --output /mnt/ngan/ISL-Sequences/manifests/isign_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CSV_DIR = Path(".csv")
DEFAULT_VIDEO_ROOT = Path("/mnt/ngan/isign_videos")
DEFAULT_OUTPUT = Path("/mnt/ngan/ISL-Sequences/manifests/isign_metadata.csv")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class Relations:
    words: set[str] = field(default_factory=set)
    word_ids: set[str] = field(default_factory=set)
    relation_types: set[str] = field(default_factory=set)
    source_csvs: set[str] = field(default_factory=set)
    example_count: int = 0
    description_count: int = 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classify_uid(uid: str) -> str:
    if uid.endswith("_w"):
        return "word"
    if uid.endswith("_d"):
        return "description"
    if re.search(r"_e\d+$", uid):
        return "example_sentence"
    if re.search(r"-\d+$", uid):
        return "general_sentence"
    return "other"


def find_video(video_root: Path, uid: str) -> Path | None:
    if not video_root.exists():
        return None

    for ext in VIDEO_EXTENSIONS:
        path = video_root / f"{uid}{ext}"
        if path.exists():
            return path

    matches = sorted(p for p in video_root.glob(f"{uid}.*") if p.is_file())
    return matches[0] if matches else None


def collect_relations(csv_dir: Path) -> dict[str, Relations]:
    rels: dict[str, Relations] = defaultdict(Relations)

    presence_path = csv_dir / "word-presence-dataset_v1.1.csv"
    if presence_path.exists():
        for row in read_csv(presence_path):
            word_id = row.get("word_id", "").strip()
            word = row.get("word", "").strip()
            sentence_id = row.get("sentence_id", "").strip()

            if word_id:
                rels[word_id].words.add(word)
                rels[word_id].word_ids.add(word_id)
                rels[word_id].relation_types.add("word")
                rels[word_id].source_csvs.add(presence_path.name)
                rels[word_id].example_count += 1

            if sentence_id:
                rels[sentence_id].words.add(word)
                rels[sentence_id].word_ids.add(word_id)
                rels[sentence_id].relation_types.add("contains_word")
                rels[sentence_id].source_csvs.add(presence_path.name)

    description_path = csv_dir / "word-description-dataset_v1.1.csv"
    if description_path.exists():
        for row in read_csv(description_path):
            word_id = row.get("word_id", "").strip()
            word = row.get("word", "").strip()
            sentence_id = row.get("sentence_id", "").strip()

            if word_id:
                rels[word_id].words.add(word)
                rels[word_id].word_ids.add(word_id)
                rels[word_id].relation_types.add("word")
                rels[word_id].source_csvs.add(description_path.name)
                rels[word_id].description_count += 1

            if sentence_id:
                rels[sentence_id].words.add(word)
                rels[sentence_id].word_ids.add(word_id)
                rels[sentence_id].relation_types.add("defines_word")
                rels[sentence_id].source_csvs.add(description_path.name)

    return rels


def join_values(values: set[str]) -> str:
    return "|".join(sorted(v for v in values if v))


def build_manifest(csv_dir: Path, video_root: Path) -> list[dict[str, str]]:
    isign_path = csv_dir / "iSign_v1.1.csv"
    if not isign_path.exists():
        raise FileNotFoundError(f"Missing required CSV: {isign_path}")

    relations = collect_relations(csv_dir)
    rows = []

    for row in read_csv(isign_path):
        uid = row.get("uid", "").strip()
        text = row.get("text", "").strip()
        category = classify_uid(uid)
        video_path = find_video(video_root, uid)
        rel = relations.get(uid, Relations())

        rows.append(
            {
                "uid": uid,
                "text": text,
                "category": category,
                "video_path": str(video_path) if video_path else "",
                "exists": "1" if video_path else "0",
                "extension": video_path.suffix if video_path else "",
                "word": join_values(rel.words),
                "word_id": join_values(rel.word_ids),
                "relation_type": join_values(rel.relation_types),
                "source_csv": join_values(rel.source_csvs),
                "example_count": str(rel.example_count),
                "description_count": str(rel.description_count),
            }
        )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "uid",
        "text",
        "category",
        "video_path",
        "exists",
        "extension",
        "word",
        "word_id",
        "relation_type",
        "source_csv",
        "example_count",
        "description_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    by_category: dict[str, int] = defaultdict(int)
    found = 0
    for row in rows:
        by_category[row["category"]] += 1
        found += row["exists"] == "1"

    print(f"Total rows       : {len(rows)}")
    print(f"Videos found     : {found}")
    print(f"Videos missing   : {len(rows) - found}")
    print("By category:")
    for category in sorted(by_category):
        print(f"  {category:18s}: {by_category[category]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_manifest(args.csv_dir, args.video_root)
    write_manifest(args.output, rows)
    print_summary(rows)
    print(f"Manifest written : {args.output}")


if __name__ == "__main__":
    main()
