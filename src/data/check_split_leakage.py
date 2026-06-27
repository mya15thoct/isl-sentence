"""Check video_id leakage across train/val/test manifests.

iSign recommends splitting by ``video_id`` (all segments of a video must stay
in one split) to avoid leakage. Our uids look like::

    <video_id>-<n>     general sentence   (e.g. 1782bea75c7d-3)
    <video_id>_e<n>    example sentence
    <video_id>_d       word description
    <video_id>_w       word

This script strips the segment suffix to recover ``video_id`` and reports any
video_id that appears in more than one split.

Usage:
    python -m src.data.check_split_leakage \
        --train .../_train.csv --val .../_val.csv --test .../_test.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

SUFFIX = re.compile(r"(_w|_d|_e\d+|-\d+)$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def video_id(row: dict[str, str]) -> str:
    """Recover the iSign video_id from a row's uid (or keypoint filename)."""
    uid = (row.get("uid") or "").strip()
    if not uid:
        kp = (row.get("keypoint_path") or "").strip()
        uid = Path(kp).stem if kp else ""
    return SUFFIX.sub("", uid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    args = ap.parse_args()

    split_vids: dict[str, set[str]] = {}
    owners: dict[str, set[str]] = defaultdict(set)  # video_id -> {splits}
    for name, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        rows = read_csv(path)
        vids = {video_id(r) for r in rows if video_id(r)}
        split_vids[name] = vids
        for v in vids:
            owners[v].add(name)
        print(f"{name:6s}: {len(rows):>7d} rows | {len(vids):>6d} unique video_ids")

    tr, va, te = split_vids["train"], split_vids["val"], split_vids["test"]
    print()
    print(f"train ∩ val  : {len(tr & va):>5d} video_ids in both")
    print(f"train ∩ test : {len(tr & te):>5d} video_ids in both")
    print(f"val   ∩ test : {len(va & te):>5d} video_ids in both")

    leaked = {v for v, s in owners.items() if len(s) > 1}
    print()
    if leaked:
        total = sum(len(v) for v in split_vids.values())
        print(f"*** LEAKAGE: {len(leaked)} video_ids span >1 split "
              f"({len(leaked) / max(1, len(owners)) * 100:.1f}% of all videos) ***")
        for v in list(leaked)[:10]:
            print(f"    {v} -> {sorted(owners[v])}")
        print("\n-> Re-split by video_id (80/10/10) is recommended for a valid benchmark.")
    else:
        print("OK: every video_id is in exactly ONE split (no leakage).")


if __name__ == "__main__":
    main()
