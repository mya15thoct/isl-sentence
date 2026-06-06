"""
Build a relation-aware iSign pretraining manifest.

The existing sentence-only pretraining uses one pair per video:

    video -> its sentence text

This script expands iSign rows into several relation-aware video/text pairs:

    sentence video     -> sentence text
    example video      -> contained word
    description video  -> defined word
    word video         -> word text
    word video         -> word description text
    word video         -> example sentence text

The output is still compatible with src/text/build_text_embeddings.py and
src/training/train_video_text_contrastive.py: every row has a keypoint_path and
a text field. Extra relation columns are preserved for audit/ablation.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


USABLE_STATUSES = {"", "ok", "skipped_exists"}


OUTPUT_FIELDS = [
    "pair_id",
    "uid",
    "text",
    "category",
    "relation_type",
    "video_uid",
    "video_category",
    "source_text",
    "word",
    "word_id",
    "sentence_uid",
    "keypoint_path",
    "num_frames",
    "extract_status",
    "quality_flag",
    "relation_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build relation-aware iSign video/text pairs for contrastive pretraining.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="QC/keypoint manifest containing uid,text,category,keypoint_path columns.",
    )
    parser.add_argument("--word-presence-csv", type=Path, required=True)
    parser.add_argument("--word-description-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-base",
        action="store_true",
        default=True,
        help="Include each video's own text as a base video/text pair.",
    )
    parser.add_argument(
        "--no-base",
        action="store_false",
        dest="include_base",
        help="Write only auxiliary relation pairs.",
    )
    parser.add_argument("--include-word-to-description", action="store_true", default=True)
    parser.add_argument("--no-word-to-description", action="store_false", dest="include_word_to_description")
    parser.add_argument("--include-word-to-example", action="store_true", default=True)
    parser.add_argument("--no-word-to-example", action="store_false", dest="include_word_to_example")
    parser.add_argument(
        "--base-categories",
        nargs="+",
        default=["general_sentence", "example_sentence", "description", "word"],
        help="Categories whose own text is used for base pairs.",
    )
    parser.add_argument("--check-keypoints", action="store_true")
    parser.add_argument("--skip-suspect", action="store_true")
    parser.add_argument("--max-word-descriptions-per-word", type=int, default=2)
    parser.add_argument("--max-word-examples-per-word", type=int, default=3)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def usable_row(row: dict[str, str], check_keypoints: bool, skip_suspect: bool) -> bool:
    if not clean_text(row.get("uid", "")):
        return False
    if not clean_text(row.get("keypoint_path", "")):
        return False
    if clean_text(row.get("extract_status", "")) not in USABLE_STATUSES:
        return False
    if skip_suspect and clean_text(row.get("quality_flag", "")) == "suspect":
        return False
    if check_keypoints and not Path(row["keypoint_path"]).is_file():
        return False
    return True


def relation_index(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        word_id = clean_text(row.get("word_id", ""))
        if word_id:
            index.setdefault(word_id, []).append(row)
    return index


def add_pair(
    output: list[dict[str, str]],
    video_row: dict[str, str],
    text: str,
    relation_type: str,
    source_text: str,
    word: str = "",
    word_id: str = "",
    sentence_uid: str = "",
    relation_weight: float = 1.0,
) -> None:
    text = clean_text(text)
    if not text:
        return

    video_uid = clean_text(video_row.get("uid", ""))
    output.append(
        {
            "pair_id": str(len(output)),
            "uid": f"{video_uid}::{relation_type}::{len(output)}",
            "text": text,
            "category": relation_type,
            "relation_type": relation_type,
            "video_uid": video_uid,
            "video_category": clean_text(video_row.get("category", "")),
            "source_text": clean_text(source_text),
            "word": clean_text(word),
            "word_id": clean_text(word_id),
            "sentence_uid": clean_text(sentence_uid),
            "keypoint_path": clean_text(video_row.get("keypoint_path", "")),
            "num_frames": clean_text(video_row.get("num_frames", "")),
            "extract_status": clean_text(video_row.get("extract_status", "")),
            "quality_flag": clean_text(video_row.get("quality_flag", "")),
            "relation_weight": f"{relation_weight:.6g}",
        }
    )


def main() -> None:
    args = parse_args()
    manifest_rows = [
        row
        for row in read_csv(args.manifest)
        if usable_row(row, check_keypoints=args.check_keypoints, skip_suspect=args.skip_suspect)
    ]
    presence_rows = read_csv(args.word_presence_csv)
    description_rows = read_csv(args.word_description_csv)

    by_uid = {clean_text(row.get("uid", "")): row for row in manifest_rows}
    word_rows = {
        clean_text(row.get("word_id", "")): row
        for row in manifest_rows
        if clean_text(row.get("category", "")) == "word" and clean_text(row.get("word_id", ""))
    }
    descriptions_by_word = relation_index(description_rows)
    examples_by_word = relation_index(presence_rows)

    output: list[dict[str, str]] = []
    base_categories = set(args.base_categories)

    if args.include_base:
        for row in manifest_rows:
            category = clean_text(row.get("category", ""))
            if category not in base_categories:
                continue
            add_pair(
                output,
                row,
                text=row.get("text", ""),
                relation_type=f"{category}_text",
                source_text=row.get("text", ""),
                word=row.get("word", ""),
                word_id=row.get("word_id", ""),
                sentence_uid=row.get("uid", ""),
                relation_weight=1.0,
            )

    for row in manifest_rows:
        category = clean_text(row.get("category", ""))
        word = clean_text(row.get("word", ""))
        word_id = clean_text(row.get("word_id", ""))
        if not word or not word_id:
            continue

        if category == "example_sentence":
            add_pair(
                output,
                row,
                text=word,
                relation_type="example_contains_word",
                source_text=row.get("text", ""),
                word=word,
                word_id=word_id,
                sentence_uid=row.get("uid", ""),
                relation_weight=0.5,
            )

        if category == "description":
            add_pair(
                output,
                row,
                text=word,
                relation_type="description_defines_word",
                source_text=row.get("text", ""),
                word=word,
                word_id=word_id,
                sentence_uid=row.get("uid", ""),
                relation_weight=0.5,
            )

    if args.include_word_to_description:
        for word_id, word_row in word_rows.items():
            for rel in descriptions_by_word.get(word_id, [])[: args.max_word_descriptions_per_word]:
                add_pair(
                    output,
                    word_row,
                    text=rel.get("sentence", ""),
                    relation_type="word_to_description",
                    source_text=word_row.get("text", ""),
                    word=word_row.get("word", "") or rel.get("word", ""),
                    word_id=word_id,
                    sentence_uid=rel.get("sentence_id", ""),
                    relation_weight=0.35,
                )

    if args.include_word_to_example:
        for word_id, word_row in word_rows.items():
            for rel in examples_by_word.get(word_id, [])[: args.max_word_examples_per_word]:
                add_pair(
                    output,
                    word_row,
                    text=rel.get("sentence", ""),
                    relation_type="word_to_example",
                    source_text=word_row.get("text", ""),
                    word=word_row.get("word", "") or rel.get("word", ""),
                    word_id=word_id,
                    sentence_uid=rel.get("sentence_id", ""),
                    relation_weight=0.25,
                )

    write_csv(args.output, output)

    relation_counts = Counter(row["relation_type"] for row in output)
    video_category_counts = Counter(row["video_category"] for row in output)
    missing_relation_uids = [
        row["sentence_id"]
        for row in presence_rows + description_rows
        if clean_text(row.get("sentence_id", "")) and clean_text(row.get("sentence_id", "")) not in by_uid
    ]

    print(f"input manifest rows : {len(manifest_rows)}")
    print(f"output pairs        : {len(output)}")
    print(f"word videos indexed : {len(word_rows)}")
    print(f"output              : {args.output}")
    print("by relation:")
    for relation, count in relation_counts.most_common():
        print(f"  {relation:28s} {count}")
    print("by video category:")
    for category, count in video_category_counts.most_common():
        print(f"  {category or '<blank>':28s} {count}")
    if missing_relation_uids:
        print(f"relation sentence ids absent from manifest: {len(set(missing_relation_uids))}")


if __name__ == "__main__":
    main()
