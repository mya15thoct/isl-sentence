from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEY_COLUMNS = [
    "keypoint_path",
    "keypoints",
    "npy_path",
    "path",
    "video_path",
    "video",
    "filename",
    "file",
    "uid",
    "id",
]
SENTENCE_COLUMNS = ["sentence", "text", "label", "class", "phrase", "utterance"]
GLOSS_COLUMNS = ["sign glosses", "sign_glosses", "glosses", "gloss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standard manifest for the 491-video target sentence set.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help="Optional row-level label CSV. If omitted, class labels are read from keypoint subdirectories.",
    )
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glosses-csv", type=Path)
    parser.add_argument("--extension", default=".npy")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def first_present(row: dict[str, str], candidates: Iterable[str]) -> str:
    normalized = {norm_name(key): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(norm_name(candidate), "")
        if value and value.strip():
            return value.strip()
    return ""


def normalize_sentence(value: str) -> str:
    value = value.strip().lower().replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def sentence_from_slug(value: str) -> str:
    return normalize_sentence(value)


def has_candidate_column(row: dict[str, str], candidates: Iterable[str]) -> bool:
    columns = {norm_name(key) for key in row}
    return any(norm_name(candidate) in columns for candidate in candidates)


def build_gloss_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}

    mapping: dict[str, str] = {}
    for row in read_csv(path):
        sentence = first_present(row, SENTENCE_COLUMNS)
        gloss = first_present(row, GLOSS_COLUMNS)
        if sentence and gloss:
            mapping[normalize_sentence(sentence)] = gloss
    return mapping


def build_sentence_name_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}

    mapping: dict[str, str] = {}
    for row in read_csv(path):
        sentence = first_present(row, SENTENCE_COLUMNS)
        if sentence:
            mapping[normalize_sentence(sentence)] = sentence.strip()
    return mapping


def key_variants(value: str) -> set[str]:
    path = Path(value)
    stem = path.stem
    name = path.name
    variants = {value, name, stem}
    if name.endswith(".npy"):
        variants.add(name[:-4])
    return {item.strip() for item in variants if item.strip()}


def index_keypoints(root: Path, extension: str) -> dict[str, Path]:
    files = sorted(root.rglob(f"*{extension}"))
    index: dict[str, Path] = {}
    for path in files:
        for key in key_variants(str(path)):
            index.setdefault(key, path)
        index.setdefault(path.name, path)
        index.setdefault(path.stem, path)
    return index


def find_keypoint(row: dict[str, str], keypoint_index: dict[str, Path]) -> tuple[Path | None, str]:
    key = first_present(row, KEY_COLUMNS)
    if not key:
        return None, "missing_key"

    key_path = Path(key)
    if key_path.is_file():
        return key_path, "direct_path"

    for variant in key_variants(key):
        match = keypoint_index.get(variant)
        if match is not None:
            return match, f"matched:{variant}"

    return None, f"not_found:{key}"


def build_from_class_dirs(
    keypoint_root: Path,
    extension: str,
    gloss_map: dict[str, str],
    sentence_name_map: dict[str, str],
) -> list[dict[str, str]]:
    class_dirs = sorted(path for path in keypoint_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class directories found under {keypoint_root}")

    output_rows: list[dict[str, str]] = []
    label_id = 0
    for class_dir in class_dirs:
        keypoint_files = sorted(class_dir.glob(f"*{extension}"))
        if not keypoint_files:
            continue

        slug_sentence = sentence_from_slug(class_dir.name)
        sentence_key = normalize_sentence(slug_sentence)
        sentence = sentence_name_map.get(sentence_key, slug_sentence)
        gloss = gloss_map.get(sentence_key, "")
        for path in keypoint_files:
            output_rows.append(
                {
                    "sample_id": str(len(output_rows)),
                    "uid": path.stem,
                    "sentence": sentence,
                    "gloss": gloss,
                    "label_id": str(label_id),
                    "label_text": sentence,
                    "keypoint_path": str(path),
                    "match_method": f"class_dir:{class_dir.name}",
                }
            )
        label_id += 1

    return output_rows


def build_from_labels_csv(
    labels_csv: Path,
    keypoint_root: Path,
    extension: str,
    gloss_map: dict[str, str],
) -> tuple[list[dict[str, str]], int]:
    rows = read_csv(labels_csv)
    if not rows:
        raise SystemExit(f"No rows found in {labels_csv}")

    keypoint_index = index_keypoints(keypoint_root, extension)
    keypoint_files = sorted({path for path in keypoint_index.values()})

    use_sorted_fallback = not has_candidate_column(rows[0], KEY_COLUMNS)
    if use_sorted_fallback and len(rows) != len(keypoint_files):
        raise SystemExit(
            "Labels CSV has no key/video column and row count does not match "
            f"keypoint files: labels={len(rows)} keypoints={len(keypoint_files)}"
        )

    output_rows: list[dict[str, str]] = []
    class_to_id: dict[str, int] = {}
    missing = 0
    for idx, row in enumerate(rows):
        sentence = first_present(row, SENTENCE_COLUMNS)
        if not sentence:
            raise SystemExit(
                f"Could not find sentence/label column. Available columns: {list(row)}"
            )

        gloss = first_present(row, GLOSS_COLUMNS) or gloss_map.get(normalize_sentence(sentence), "")
        if use_sorted_fallback:
            keypoint_path = keypoint_files[idx]
            match_method = "sorted_order"
        else:
            keypoint_path, match_method = find_keypoint(row, keypoint_index)
            if keypoint_path is None:
                missing += 1
                continue

        label_id = class_to_id.setdefault(sentence, len(class_to_id))
        output_rows.append(
            {
                "sample_id": str(idx),
                "uid": keypoint_path.stem,
                "sentence": sentence,
                "gloss": gloss,
                "label_id": str(label_id),
                "label_text": sentence,
                "keypoint_path": str(keypoint_path),
                "match_method": match_method,
            }
        )

    return output_rows, missing


def main() -> None:
    args = parse_args()
    gloss_map = build_gloss_map(args.glosses_csv)
    sentence_name_map = build_sentence_name_map(args.glosses_csv)
    missing = 0
    if args.labels_csv is None:
        output_rows = build_from_class_dirs(
            args.keypoint_root,
            args.extension,
            gloss_map,
            sentence_name_map,
        )
    else:
        output_rows, missing = build_from_labels_csv(
            args.labels_csv,
            args.keypoint_root,
            args.extension,
            gloss_map,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "uid",
        "sentence",
        "gloss",
        "label_id",
        "label_text",
        "keypoint_path",
        "match_method",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    class_count = len({row["label_id"] for row in output_rows})
    missing_gloss = sum(1 for row in output_rows if not row.get("gloss"))
    if args.labels_csv is not None:
        print(f"labels csv      : {args.labels_csv}")
    else:
        print("labels source   : keypoint class directories")
    print(f"keypoint root   : {args.keypoint_root}")
    print(f"manifest rows   : {len(output_rows)}")
    print(f"classes         : {class_count}")
    print(f"missing keypoint: {missing}")
    print(f"missing gloss   : {missing_gloss}")
    print(f"output          : {args.output}")


if __name__ == "__main__":
    main()
