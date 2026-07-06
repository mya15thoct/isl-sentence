#!/usr/bin/env bash
# Download How2Sign (Green Screen RGB CLIPS, frontal + realigned English CSVs).
#
# The files are served from Google Drive links listed on https://how2sign.github.io/
# (CC BY-NC 4.0 — research use only; do not redistribute). Google Drive links
# occasionally rotate, so: open the site, copy the share links for
#   - "Green Screen RGB clips* (frontal view)"  train / val / test
#   - "English Translation (manually re-aligned)" train / val / test
# and paste their FILE IDs below, then run this script.
#
#   pip install gdown            # once
#   bash cross_dataset/download_how2sign.sh
set -euo pipefail

ROOT=${ROOT:-/mnt/recover/ngan/How2Sign}
mkdir -p "$ROOT/zips" "$ROOT/text" "$ROOT/clips_train" "$ROOT/clips_val" "$ROOT/clips_test"

# ---- paste Google Drive FILE IDs from https://how2sign.github.io/ ----
ID_CLIPS_TRAIN=""   # Green Screen RGB clips (frontal), train  (~31 GB)
ID_CLIPS_VAL=""     # val   (~1.7 GB)
ID_CLIPS_TEST=""    # test  (~2.2 GB)
ID_CSV_TRAIN=""     # how2sign_realigned_train.csv
ID_CSV_VAL=""       # how2sign_realigned_val.csv
ID_CSV_TEST=""      # how2sign_realigned_test.csv
# ----------------------------------------------------------------------

for v in ID_CLIPS_TRAIN ID_CLIPS_VAL ID_CLIPS_TEST ID_CSV_TRAIN ID_CSV_VAL ID_CSV_TEST; do
  if [ -z "${!v}" ]; then
    echo "!! $v is empty — open https://how2sign.github.io/, copy the Drive link, paste its file id."
    exit 1
  fi
done

echo "== CSVs =="
gdown "$ID_CSV_TRAIN" -O "$ROOT/text/how2sign_realigned_train.csv"
gdown "$ID_CSV_VAL"   -O "$ROOT/text/how2sign_realigned_val.csv"
gdown "$ID_CSV_TEST"  -O "$ROOT/text/how2sign_realigned_test.csv"

echo "== clips: val + test first (small), then train (31 GB) =="
gdown "$ID_CLIPS_VAL"  -O "$ROOT/zips/clips_val.zip"
gdown "$ID_CLIPS_TEST" -O "$ROOT/zips/clips_test.zip"
gdown "$ID_CLIPS_TRAIN" -O "$ROOT/zips/clips_train.zip"

# Unzip flat into per-split clip folders, then drop each zip to save disk.
for s in val test train; do
  echo "== unzip $s =="
  unzip -q -j "$ROOT/zips/clips_$s.zip" -d "$ROOT/clips_$s"
  rm -f "$ROOT/zips/clips_$s.zip"
done
echo "done. next: extract_how2sign.py (use --delete-after to free clips as you go)"
