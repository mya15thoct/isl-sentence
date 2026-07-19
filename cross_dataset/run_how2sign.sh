#!/usr/bin/env bash
# How2Sign cross-dataset runs: HARP + the uniform-fusion baseline, official split.
# Same recipe as the iSign headline (fairness item 5); no per-dataset tuning.
# Evaluate on the FULL official test pool (pool-sizes 0) — the CiCo/SEDS protocol.
set -euo pipefail

ROOT=${ROOT:-/mnt/recover/ngan/How2Sign}
TRAIN=$ROOT/manifests/how2sign_train.csv
VAL=$ROOT/manifests/how2sign_val.csv
TEST=$ROOT/manifests/how2sign_test.csv
CKPT=$ROOT/checkpoints
LOG=$ROOT/logs
BGE="BAAI/bge-large-en-v1.5"
mkdir -p "$LOG" "$ROOT/eval"

COMMON="--manifest $TRAIN --val-manifest $VAL --text-column canonical_text \
  --epochs 50 --batch-size 128 --num-workers 4 --max-frames 512 \
  --sample-mode random --augment --augment-prob 0.8 \
  --lr 2e-4 --text-lr 1e-5 --warmup-epochs 2 --density-weight 0 \
  --ema-decay 0.999 --seed 42 --device cuda --amp --print-every 50"

run () {
  local name=$1; shift
  if [ -f "$CKPT/h2s_$name/checkpoint_ema.pt" ]; then
    echo "=== SKIP: $name ==="; return 0
  fi
  echo "=== RUN: $name ==="
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       python -u src/retrieval/train.py $COMMON \
       --save-dir "$CKPT/h2s_$name" "$@" > "$LOG/h2s_$name.log" 2>&1; then
    echo "=== DONE: $name ==="
  else
    echo "=== FAILED: $name (see $LOG/h2s_$name.log) ==="
  fi
}

# Three of our rows, mirroring the iSign ladder: frozen-text -> uniform-fusion
# (SignCLIP-style) -> HARP. The transfer claim needs the ladder, not one number.
run harp         --text-model "$BGE" --embedding-dim 1024 --hand-aware
run no_handaware --text-model "$BGE" --embedding-dim 1024
run frozen_text  --text-model "$BGE" --embedding-dim 1024 --hand-aware --text-lr 0

echo "=== EVAL (full official test pool) ==="
for d in harp no_handaware frozen_text; do
  python -m src.retrieval.evaluate_rerank \
    --manifest "$TEST" \
    --checkpoints "$CKPT/h2s_$d/checkpoint_ema.pt" \
    --pool-sizes 0 --out "$ROOT/eval/h2s_$d.json" \
    --dump-embeddings "$ROOT/eval/emb_h2s_$d.pt"
done

# The transfer claim is the hand-aware vs uniform GAP replicating on ASL —
# significance-test it the same way as on iSign (paired bootstrap, 95% CI).
python -m src.retrieval.bootstrap_compare \
  --dump-a "$ROOT/eval/emb_h2s_harp.pt" --dump-b "$ROOT/eval/emb_h2s_no_handaware.pt" \
  --label-a HARP --label-b uniform-fusion --out "$ROOT/eval/boot_h2s_handaware.json"

echo "Report BOTH cosine and re-ranked numbers next to CiCo/SEDS (see README)."
