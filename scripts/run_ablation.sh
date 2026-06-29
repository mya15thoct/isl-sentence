#!/usr/bin/env bash
# Clean ablation for the iSign pose-text retrieval paper.
#
# Every row uses the SAME training protocol (config + seed below); each variant
# toggles exactly ONE component versus the reference, so differences are
# attributable to that component alone. Run sequentially (one GPU).
#
# After training, score every checkpoint with the SAME evaluator:
#   see scripts note at the bottom (evaluate_rerank over all checkpoint_best.pt).
set -euo pipefail

# ---- paths (edit if needed) ----
ROOT=/mnt/ngan/ISL-Sequences
# video_id-grouped 80/10/10 split (leakage-free, iSign protocol)
TRAIN=$ROOT/manifests/isign_retrieval_videosplit_train.csv
VAL=$ROOT/manifests/isign_retrieval_videosplit_val.csv
CKPT=$ROOT/checkpoints
LOG=$ROOT/logs
BGE="BAAI/bge-large-en-v1.5"
MINILM="sentence-transformers/all-MiniLM-L6-v2"
mkdir -p "$LOG"

# ---- shared protocol (identical for every row) ----
# EXACTLY the abl_ema50 headline recipe (50 ep, batch 128, EMA 0.999, seed 42), so
# the reference row IS the already-trained abl_ema50 checkpoint and every variant
# below differs from it by exactly ONE component. Do NOT re-run the reference.
COMMON="--manifest $TRAIN --val-manifest $VAL --text-column canonical_text \
  --epochs 50 --batch-size 128 --num-workers 4 --max-frames 512 \
  --sample-mode random --augment --augment-prob 0.8 \
  --lr 2e-4 --text-lr 1e-5 --warmup-epochs 2 --density-weight 0 \
  --ema-decay 0.999 --seed 42 --device cuda --amp --print-every 50"

run () {  # run <name> <extra-flags...>
  local name=$1; shift
  echo "=== RUN: $name ==="
  # Tolerate a single row failing (e.g. transient OOM) so the rest still run.
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       python -u src/retrieval/train.py $COMMON \
       --save-dir "$CKPT/abl_$name" "$@" \
       > "$LOG/abl_$name.log" 2>&1; then
    echo "=== DONE: $name (in $CKPT/abl_$name) ==="
  else
    echo "=== FAILED: $name (see $LOG/abl_$name.log) — continuing ==="
  fi
}

# ---- ablation rows (toggle ONE thing vs the reference) ----
# REFERENCE = $CKPT/abl_ema50 (hand-aware + bge + redundancy, full context).
# Already trained — NOT re-run here.

# − hand-aware  (vanilla pose encoder = SignCLIP-style baseline on iSign)
run no_handaware  --text-model "$BGE"    --embedding-dim 1024

# − redundancy grouping (#1 off: identical captions become hard negatives)
run no_redundancy --text-model "$BGE"    --embedding-dim 1024 --hand-aware --no-redundancy

# − bge  (weaker text encoder: MiniLM instead of bge-large)
run minilm_text   --text-model "$MINILM" --embedding-dim 384  --hand-aware

# ---- INPUT-STREAM ablation (proves the hand-aware design) ----
# All use hand-aware + bge; vary which parts feed the cross-attention context.
# hands only  (no context)
run ctx_hands_only --text-model "$BGE" --embedding-dim 1024 --hand-aware --context-parts
# hands + pose
run ctx_hands_pose --text-model "$BGE" --embedding-dim 1024 --hand-aware --context-parts pose
# hands + face
run ctx_hands_face --text-model "$BGE" --embedding-dim 1024 --hand-aware --context-parts face
# hands + pose + face = the `reference` row above (full context)

echo "ALL ABLATION RUNS COMPLETE."
echo "Next — score every row (incl. the ema50 reference) with the SAME evaluator,"
echo "using checkpoint_ema.pt and the TEST split (Sinkhorn is reported alongside"
echo "cosine/DSL automatically):"
echo "  for d in ema50 no_handaware no_redundancy minilm_text ctx_hands_only ctx_hands_pose ctx_hands_face; do"
echo "    python -m src.retrieval.evaluate_rerank \\"
echo "      --manifest $ROOT/manifests/isign_retrieval_videosplit_test.csv \\"
echo "      --checkpoints $CKPT/abl_\$d/checkpoint_ema.pt \\"
echo "      --pool-sizes 0 2000 1000 --out $ROOT/eval/abl_\$d.json; done"
