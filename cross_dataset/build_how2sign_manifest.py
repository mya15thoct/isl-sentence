"""Build retrieval manifests for How2Sign from the official (re-aligned) CSVs.

Keeps the OFFICIAL split untouched (fairness item 1): every CSV row of a split
goes into that split's manifest; rows whose keypoints are missing/failed are
written to a sidecar `_missing.csv` and REPORTED, never silently dropped from
the test pool accounting.

Captions are used as-is (whitespace-trimmed only) — no iSign-style cleaning
(fairness item 2). Output columns match the iSign manifests so `train.py` /
`evaluate_rerank.py` work unchanged: uid, canonical_text, keypoint_path.

    python cross_dataset/build_how2sign_manifest.py \
        --csv-dir /mnt/recover/ngan/How2Sign/text \
        --keypoint-root /mnt/recover/ngan/How2Sign/keypoints \
        --out-dir /mnt/recover/ngan/How2Sign/manifests
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

SPLITS = {  # official How2Sign re-aligned CSVs (tab-separated)
    "train": "how2sign_realigned_train.csv",
    "val": "how2sign_realigned_val.csv",
    "test": "how2sign_realigned_test.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--keypoint-root", type=Path, required=True,
                    help="contains train/ val/ test/ subfolders of .npy")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, csv_name in SPLITS.items():
        src = args.csv_dir / csv_name
        rows_out, missing = [], []
        with src.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                name = (row.get("SENTENCE_NAME") or "").strip()
                text = " ".join((row.get("SENTENCE") or "").split())
                if not name or not text:
                    continue
                kp = args.keypoint_root / split / f"{name}.npy"
                rec = {"uid": name, "canonical_text": text, "keypoint_path": str(kp)}
                (rows_out if kp.exists() else missing).append(rec)

        out = args.out_dir / f"how2sign_{split}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["uid", "canonical_text", "keypoint_path"])
            writer.writeheader()
            writer.writerows(rows_out)
        if missing:
            miss = args.out_dir / f"how2sign_{split}_missing.csv"
            with miss.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["uid", "canonical_text", "keypoint_path"])
                writer.writeheader()
                writer.writerows(missing)
        total = len(rows_out) + len(missing)
        print(f"{split:5s}: {len(rows_out):>6d} usable / {total:>6d} official rows"
              f"  ({len(missing)} missing keypoints{' -> ' + str(miss) if missing else ''})")
        if split == "test" and missing:
            print(f"  !! disclose in the paper: {len(missing)}/{total} test rows lack keypoints; "
                  f"count them as failures (rank=inf), do NOT shrink the official pool.")


if __name__ == "__main__":
    main()
