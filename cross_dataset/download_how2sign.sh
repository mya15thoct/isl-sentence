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

# ---- paste from https://how2sign.github.io/ (right-click each button -> Copy link address) ----
# Train clips are a Drive FOLDER (not a single zip) -- paste the folder ID from
# .../drive/folders/<ID>. The other five are single files -- paste the ID from
# .../file/d/<ID>/view.
FOLDER_CLIPS_TRAIN="16NGheDRYNdvOxQKV_ty7zOwPjgfvPJQz"
ID_CLIPS_VAL="1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g"
ID_CLIPS_TEST="1z0i6BBGHQ12ChY63hZH56QnczvQ0JfTb"
ID_CSV_TRAIN="1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok"
ID_CSV_VAL="1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI"
ID_CSV_TEST="1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu"
# ----------------------------------------------------------------------------

for v in FOLDER_CLIPS_TRAIN ID_CLIPS_VAL ID_CLIPS_TEST ID_CSV_TRAIN ID_CSV_VAL ID_CSV_TEST; do
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

echo "== clips: val + test first (small, single-file zips) =="
for s in val test; do
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

echo "== clips: train (Drive FOLDER, ~31 GB, may be split across multiple files) =="
if [ -n "$(ls -A "$ROOT/clips_train" 2>/dev/null)" ]; then
  echo "== clips_train already present — skip download =="
else
gdown --folder "https://drive.google.com/drive/folders/$FOLDER_CLIPS_TRAIN" -O "$ROOT/clips_train_raw"
# The folder may contain one or several zip/tar parts, or already-extracted
# videos. Handle both without guessing wrong:
shopt -s nullglob
archives=("$ROOT"/clips_train_raw/*.zip)
if [ ${#archives[@]} -gt 0 ]; then
  for a in "${archives[@]}"; do
    echo "== unzip $(basename "$a") =="
    unzip -q -j "$a" -d "$ROOT/clips_train"
    rm -f "$a"
  done
else
  echo "no .zip found in clips_train_raw — assuming videos are already extracted; moving them"
  find "$ROOT/clips_train_raw" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' \) \
    -exec mv -t "$ROOT/clips_train" {} +
fi
rmdir "$ROOT/clips_train_raw" 2>/dev/null || echo "note: $ROOT/clips_train_raw not empty, check leftovers"
fi

echo "done. next: extract_how2sign.py (use --delete-after to free clips as you go)"
