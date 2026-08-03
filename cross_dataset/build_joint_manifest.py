"""Combine iSign (ISL) and How2Sign (ASL) manifests into one multilingual manifest.

For the cross-lingual study: train a single pose-text model on ISL + ASL jointly
(both have English captions, so the text space is shared) and ask whether joint
training transfers — and whether the hand-aware design transfers better than
uniform fusion. This script unifies the two manifests to a common schema
(``uid, canonical_text, keypoint_path, source_dataset``), auto-detecting the text
and keypoint columns of each. Build the train and val joins separately:

    python cross_dataset/build_joint_manifest.py \
        --manifests /mnt/.../isign_retrieval_videosplit_train.csv /mnt/.../how2sign_train.csv \
        --labels isign how2sign --out /mnt/.../joint_train.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval.dataset import read_csv

TEXT_CANDIDATES = ("canonical_text", "text", "caption", "sentence", "translation")
KEYPOINT_CANDIDATES = ("keypoint_path", "npy_path", "keypoints", "pose_path")


def detect(cols: list[str], candidates: tuple[str, ...], override: str | None) -> str:
    if override:
        if override not in cols:
            raise SystemExit(f"column {override!r} not in manifest (have: {cols})")
        return override
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"could not auto-detect a column among {candidates}; columns are {cols}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", type=Path, nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True, help="dataset tag per manifest (e.g. isign how2sign)")
    p.add_argument("--text-columns", nargs="*", help="override text column per manifest (else auto-detect)")
    p.add_argument("--keypoint-columns", nargs="*", help="override keypoint column per manifest")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.manifests):
        raise SystemExit("--labels must match --manifests")
    tcols = args.text_columns or [None] * len(args.manifests)
    kcols = args.keypoint_columns or [None] * len(args.manifests)

    out_rows: list[dict[str, str]] = []
    for manifest, label, tc, kc in zip(args.manifests, args.labels, tcols, kcols):
        rows = read_csv(manifest)
        if not rows:
            print(f"!! {label}: empty manifest, skipped"); continue
        cols = list(rows[0].keys())
        text_col = detect(cols, TEXT_CANDIDATES, tc)
        kp_col = detect(cols, KEYPOINT_CANDIDATES, kc)
        kept = 0
        for i, r in enumerate(rows):
            text = (r.get(text_col) or "").strip()
            kp = (r.get(kp_col) or "").strip()
            if not text or not kp:
                continue
            out_rows.append({
                "uid": r.get("uid", "") or f"{label}_{i}",
                "canonical_text": text,
                "keypoint_path": kp,
                "source_dataset": label,
            })
            kept += 1
        print(f"{label}: {kept}/{len(rows)} rows  (text={text_col}, keypoint={kp_col})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "canonical_text", "keypoint_path", "source_dataset"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows -> {args.out}")
    by = {}
    for r in out_rows:
        by[r["source_dataset"]] = by.get(r["source_dataset"], 0) + 1
    print("composition:", by)


if __name__ == "__main__":
    main()
