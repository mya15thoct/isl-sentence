"""
Extract MediaPipe Holistic keypoints from videos listed in an iSign manifest.

Recommended order:
  1. words
  2. examples
  3. descriptions
  4. general sentences

Example on server:
  python src/video/extract_from_manifest.py \
      --manifest data/subsets/isign_words.csv \
      --video-root /mnt/ngan/isign_videos \
      --output-root /mnt/ngan/ISL-Sequences/isign_keypoints \
      --status-output /mnt/ngan/ISL-Sequences/manifests/isign_words_keypoints.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from src.keypoints.holistic import (
        create_holistic_model,
        extract_video_keypoints,
    )
except ImportError:
    from holistic import create_holistic_model, extract_video_keypoints


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


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


def find_video(video_root: Path, uid: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        path = video_root / f"{uid}{ext}"
        if path.exists():
            return path
    matches = sorted(p for p in video_root.glob(f"{uid}.*") if p.is_file())
    return matches[0] if matches else None


def output_path_for(row: dict[str, str], output_root: Path) -> Path:
    category = row.get("category") or "unknown"
    uid = row["uid"]
    return output_root / category / f"{uid}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_csv(args.manifest)

    if args.start:
        rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    status_fields = list(fieldnames)
    for field in ("keypoint_path", "num_frames", "extract_status", "extract_error"):
        if field not in status_fields:
            status_fields.append(field)

    holistic = create_holistic_model(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    processed: list[dict[str, str]] = []
    ok_count = 0
    skip_count = 0
    fail_count = 0

    with holistic:
        for idx, row in enumerate(rows, start=1 + args.start):
            uid = row["uid"]
            raw_video_path = row.get("video_path", "").strip()
            video_path = Path(raw_video_path) if raw_video_path else None
            if video_path is None or not video_path.is_file():
                video_path = find_video(args.video_root, uid)

            out_path = output_path_for(row, args.output_root)
            row = dict(row)
            row["video_path"] = str(video_path) if video_path else ""
            row["keypoint_path"] = str(out_path)
            row["num_frames"] = ""
            row["extract_status"] = ""
            row["extract_error"] = ""

            if out_path.exists() and not args.overwrite:
                try:
                    existing = np.load(out_path, mmap_mode="r")
                    row["num_frames"] = str(existing.shape[0])
                    row["extract_status"] = "skipped_exists"
                    processed.append(row)
                    skip_count += 1
                    continue
                except Exception as exc:
                    print(f"Unreadable existing keypoints, re-extracting: {out_path} ({exc})")

            if video_path is None or not video_path.is_file():
                row["extract_status"] = "missing_video"
                row["extract_error"] = f"video not found for uid={uid}"
                processed.append(row)
                fail_count += 1
                continue

            try:
                keypoints = extract_video_keypoints(video_path, holistic)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, keypoints)
                row["num_frames"] = str(keypoints.shape[0])
                row["extract_status"] = "ok"
                ok_count += 1
            except Exception as exc:
                row["extract_status"] = "error"
                row["extract_error"] = str(exc)
                fail_count += 1

            processed.append(row)

            if idx % 25 == 0:
                print(
                    f"[{idx}] ok={ok_count} skipped={skip_count} failed={fail_count}"
                )

    status_output = args.status_output or (
        args.manifest.parent / f"{args.manifest.stem}_keypoints.csv"
    )
    write_csv(status_output, status_fields, processed)

    print(f"Rows processed : {len(processed)}")
    print(f"OK             : {ok_count}")
    print(f"Skipped        : {skip_count}")
    print(f"Failed         : {fail_count}")
    print(f"Status CSV     : {status_output}")


if __name__ == "__main__":
    main()
