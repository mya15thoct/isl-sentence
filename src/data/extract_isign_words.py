"""
Extract iSign word videos into a separate folder.

The iSign CSVs contain word-level video ids ending with "_w".
This script collects those ids, finds the matching video files in the
iSign video root, and copies or symlinks them into a word-only folder.

Default behavior is dry-run.

Example:
  python src/data/extract_isign_words.py \
      --video-root /mnt/ngan/isign_videos \
      --output-root /mnt/ngan/isign_words \
      --mode copy \
      --execute
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV_DIR = Path(".csv")
DEFAULT_VIDEO_ROOT = Path("/mnt/ngan/isign_videos")
DEFAULT_OUTPUT_ROOT = Path("/mnt/ngan/isign_words")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass(frozen=True)
class WordEntry:
    uid: str
    word: str
    source: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def collect_word_entries(csv_dir: Path) -> dict[str, WordEntry]:
    entries: dict[str, WordEntry] = {}

    isign_path = csv_dir / "iSign_v1.1.csv"
    if isign_path.exists():
        for row in read_csv(isign_path):
            uid = row.get("uid", "").strip()
            if uid.endswith("_w"):
                entries.setdefault(
                    uid,
                    WordEntry(uid=uid, word=row.get("text", "").strip(), source="iSign"),
                )

    for filename in (
        "word-presence-dataset_v1.1.csv",
        "word-description-dataset_v1.1.csv",
    ):
        path = csv_dir / filename
        if not path.exists():
            continue
        for row in read_csv(path):
            uid = row.get("word_id", "").strip()
            if uid.endswith("_w"):
                entries[uid] = WordEntry(
                    uid=uid,
                    word=row.get("word", "").strip(),
                    source=filename,
                )

    return dict(sorted(entries.items()))


def find_video(video_root: Path, uid: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        path = video_root / f"{uid}{ext}"
        if path.exists():
            return path

    matches = list(video_root.glob(f"{uid}.*"))
    files = [p for p in matches if p.is_file()]
    if files:
        return sorted(files)[0]
    return None


def safe_filename(value: str) -> str:
    keep = []
    for ch in value.strip():
        if ch.isalnum() or ch in ("-", "_", " "):
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip().replace(" ", "_")
    return name or "unknown"


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["uid", "word", "source", "video_path", "output_path", "status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "hardlink"),
        default="copy",
        help="How to place files in output-root.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Save as output-root/{uid}{ext}; otherwise output-root/{word}/{uid}{ext}.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually create files. Without this flag, only writes manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path. Default: output-root/isign_words_manifest.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or (args.output_root / "isign_words_manifest.csv")

    entries = collect_word_entries(args.csv_dir)
    rows: list[dict[str, str]] = []
    found = missing = created = 0

    for entry in entries.values():
        src = find_video(args.video_root, entry.uid)
        if src is None:
            missing += 1
            rows.append(
                {
                    "uid": entry.uid,
                    "word": entry.word,
                    "source": entry.source,
                    "video_path": "",
                    "output_path": "",
                    "status": "missing",
                }
            )
            continue

        found += 1
        if args.flat:
            dst = args.output_root / f"{entry.uid}{src.suffix}"
        else:
            dst = args.output_root / safe_filename(entry.word) / f"{entry.uid}{src.suffix}"

        status = "dry_run"
        if args.execute:
            link_or_copy(src, dst, args.mode)
            created += 1
            status = args.mode

        rows.append(
            {
                "uid": entry.uid,
                "word": entry.word,
                "source": entry.source,
                "video_path": str(src),
                "output_path": str(dst),
                "status": status,
            }
        )

    write_manifest(manifest_path, rows)

    print(f"Collected word ids : {len(entries)}")
    print(f"Found videos       : {found}")
    print(f"Missing videos     : {missing}")
    print(f"Manifest           : {manifest_path}")
    if args.execute:
        print(f"Created ({args.mode})    : {created}")
        print(f"Output root        : {args.output_root}")
    else:
        print("Dry run only. Re-run with --execute to create files.")


if __name__ == "__main__":
    main()
