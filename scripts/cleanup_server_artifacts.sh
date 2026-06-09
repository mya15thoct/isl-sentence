#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/ngan/ISL-Sequences}"
MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

case "$ROOT" in
  /mnt/ngan/ISL-Sequences) ;;
  /mnt/ngan/ISL-Sequences/) ROOT="${ROOT%/}" ;;
  *)
    echo "refusing unexpected ROOT: $ROOT" >&2
    echo "set ROOT=/mnt/ngan/ISL-Sequences or edit this script intentionally" >&2
    exit 2
    ;;
esac

protected_paths=(
  "$ROOT/isign_keypoints"
  "$ROOT/manifests/isign_existing_clean_20260603_1540.csv"
  "$ROOT/manifests/isign_train_qc_20260603_1540.csv"
  "$ROOT/manifests/isign_sentence_train_qc_20260603_1540.csv"
  "$ROOT/manifests/isign_word_train_qc_20260603_1540.csv"
  "$ROOT/manifests/isign_suspect_qc_20260603_1540.csv"
  "$ROOT/manifests/isign_reject_qc_20260603_1540.csv"
)

is_protected() {
  local candidate="$1"
  local resolved_candidate resolved_protected
  resolved_candidate="$(readlink -f "$candidate" 2>/dev/null || true)"
  [[ -n "$resolved_candidate" ]] || return 1

  for protected in "${protected_paths[@]}"; do
    resolved_protected="$(readlink -f "$protected" 2>/dev/null || true)"
    [[ -n "$resolved_protected" ]] || continue
    if [[ "$resolved_candidate" == "$resolved_protected" || "$resolved_candidate" == "$resolved_protected/"* ]]; then
      return 0
    fi
  done
  return 1
}

collect_candidates() {
  find "$ROOT/checkpoints" -maxdepth 1 \
    \( -name "target_*" \
       -o -name "*prototype*" \
       -o -name "*resume*" \
       -o -name "*relation_aware*" \
       -o -name "isign_relation_*" \
       -o -name "signpose2text_*" \) \
    -print 2>/dev/null || true

  # Generation (signpose2text) is retired; drop its TRAINING logs only.
  # pred_signpose2text_*.csv and eval_signpose2text_*.json are kept on purpose
  # as the reported generation baseline for the paper.
  find "$ROOT/logs" -maxdepth 1 \
    \( -name "train_target_*.log" \
       -o -name "*resume*.log" \
       -o -name "*prototype*" \
       -o -name "eval_retrieval_*" \
       -o -name "train_isign_relation_*.log" \
       -o -name "train_signpose2text_*.log" \) \
    -print 2>/dev/null || true

  find "$ROOT/text_embeddings" -maxdepth 1 \
    \( -name "target_*" \
       -o -name "isign_relation_*" \) \
    -print 2>/dev/null || true

  find "$ROOT/manifests" -maxdepth 1 \
    \( -name "target_manifest*.csv" \
       -o -name "isign_relation_*.csv" \) \
    -print 2>/dev/null || true
}

echo "ROOT=$ROOT"
echo "MODE=$MODE"
echo

echo "Protected keypoint count:"
if [[ -d "$ROOT/isign_keypoints" ]]; then
  find "$ROOT/isign_keypoints" -name "*.npy" | wc -l
else
  echo "missing: $ROOT/isign_keypoints" >&2
  exit 1
fi
echo

mapfile -t candidates < <(collect_candidates | sort -u)

if [[ "${#candidates[@]}" -eq 0 ]]; then
  echo "No cleanup candidates found."
  exit 0
fi

echo "Cleanup candidates:"
printf '%s\n' "${candidates[@]}"
echo

for candidate in "${candidates[@]}"; do
  if is_protected "$candidate"; then
    echo "refusing protected candidate: $candidate" >&2
    exit 1
  fi
done

if [[ "$MODE" == "--dry-run" ]]; then
  echo "Dry run only. Re-run with --execute to delete these candidates."
  exit 0
fi

echo "Deleting ${#candidates[@]} candidates..."
for candidate in "${candidates[@]}"; do
  rm -rf -- "$candidate"
done

echo "Post-cleanup keypoint count:"
find "$ROOT/isign_keypoints" -name "*.npy" | wc -l

echo "Protected manifests:"
ls -lh "${protected_paths[@]:1}"
