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

# ---- Drive file IDs from https://how2sign.github.io/ ------------------------
# VERIFIED against Drive metadata (2026-07-19): names + sizes match the site's
# "Green Screen RGB clips (frontal view)" sentence-level CLIPS:
#   train_rgb_front_clips.zip 33.0 GB / val 1.78 GB / test 2.41 GB.
# (Do NOT use the *_raw_videos.zip links — those are uncut raw videos.)
ID_CLIPS_TRAIN="1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW"
ID_CLIPS_VAL="1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO"
ID_CLIPS_TEST="1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG"
ID_CSV_TRAIN="1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok"
ID_CSV_VAL="1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI"
ID_CSV_TEST="1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu"
# ----------------------------------------------------------------------------

for v in ID_CLIPS_TRAIN ID_CLIPS_VAL ID_CLIPS_TEST ID_CSV_TRAIN ID_CSV_VAL ID_CSV_TEST; do
  if [ -z "${!v}" ]; then
    echo "!! $v is empty — open https://how2sign.github.io/, copy the Drive link, paste its id."
    exit 1
  fi
done

# Google Drive rate-limits shared files ("Too many users have viewed or
# downloaded this file recently", resets within ~24 h). Every step below skips
# what already exists, so just RE-RUN this script later to fetch the rest.

echo "== CSVs =="
[ -s "$ROOT/text/how2sign_realigned_train.csv" ] || gdown "$ID_CSV_TRAIN" -O "$ROOT/text/how2sign_realigned_train.csv"
[ -s "$ROOT/text/how2sign_realigned_val.csv" ]   || gdown "$ID_CSV_VAL"   -O "$ROOT/text/how2sign_realigned_val.csv"
[ -s "$ROOT/text/how2sign_realigned_test.csv" ]  || gdown "$ID_CSV_TEST"  -O "$ROOT/text/how2sign_realigned_test.csv"

echo "== clips: val + test first (small), then train (33 GB) =="
for s in val test train; do
  if [ -n "$(ls -A "$ROOT/clips_$s" 2>/dev/null)" ]; then
    echo "== clips_$s already present — skip =="
    continue
  fi
  id_var="ID_CLIPS_$(echo "$s" | tr '[:lower:]' '[:upper:]')"
  gdown "${!id_var}" -O "$ROOT/zips/clips_$s.zip"
  echo "== unzip $s =="
  unzip -q -j "$ROOT/zips/clips_$s.zip" -d "$ROOT/clips_$s"
  rm -f "$ROOT/zips/clips_$s.zip"
done

echo "done. next: extract_how2sign.py (use --delete-after to free clips as you go)"
