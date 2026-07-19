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

# ---- paths (override with `ROOT=... bash scripts/run_ablation.sh` if remounted) ----
ROOT=${ROOT:-/mnt/recover/ngan/ISL-Sequences}
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
  # Resume-friendly: skip a row that already produced its EMA checkpoint.
  if [ -f "$CKPT/abl_$name/checkpoint_ema.pt" ]; then
    echo "=== SKIP: $name (checkpoint_ema.pt already exists) ==="
    return 0
  fi
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

# ---- REVIEW-FIX rows (Q1 revision: baselines, seeds, design-choice ablations) ----
# NOTE: later flags override $COMMON (argparse keeps the last occurrence), so
# e.g. `--seed 43` after $COMMON takes effect.

# (Issue 4) 2 extra seeds for the DECISIVE baseline. The reference already has
# 3 seeds (abl_ema50{,_s43,_s44}); the uniform-fusion baseline gets the same
# treatment so hand-aware vs uniform is mean±std on BOTH sides.
run no_handaware_s43 --text-model "$BGE" --embedding-dim 1024 --seed 43
run no_handaware_s44 --text-model "$BGE" --embedding-dim 1024 --seed 44

# (Issue 1) frozen text encoder — completes the frozen -> uniform -> HARP
# ladder on iSign, mirroring cross_dataset/run_how2sign.sh.
run frozen_text --text-model "$BGE" --embedding-dim 1024 --hand-aware --text-lr 0

# (Issue 1) CLIP4Clip-meanP-style temporal dual encoder: uniform frame MLP,
# NO Conformer blocks, temporal mean pooling — an independent baseline family.
run clip4clip_meanp --text-model "$BGE" --embedding-dim 1024 --pose-layers 0 --pose-pooling mean

# (Issue 10 / defense Q7) CLS vs mean pooling for bge-large.
run cls_pool --text-model "$BGE" --embedding-dim 1024 --hand-aware --text-pooling cls

# (Issue 10) semantic-positive threshold sensitivity (reference uses 0.85).
run sem080 --text-model "$BGE" --embedding-dim 1024 --hand-aware --semantic-threshold 0.80
run sem090 --text-model "$BGE" --embedding-dim 1024 --hand-aware --semantic-threshold 0.90

# (Issue 1.3) closest SignCLIP-style reproduction on iSign: single linear frame
# projection (no part structure) + vanilla Transformer + temporal mean pooling.
run signclip_style --text-model "$BGE" --embedding-dim 1024 --frame-encoder linear --temporal transformer --pose-pooling mean

# (Issue 1.4) simple input baselines on the uniform encoder: which body parts
# does the task actually need? (input stays 1662-d; unused parts zeroed)
run pose_only --text-model "$BGE" --embedding-dim 1024 --input-parts pose
run face_free --text-model "$BGE" --embedding-dim 1024 --input-parts pose hands

# (Issue 10) reduced-face landmarks (lips/eyebrows/eyes only) on the reference.
run face_keep --text-model "$BGE" --embedding-dim 1024 --hand-aware --face-keep

echo "ALL ABLATION RUNS COMPLETE."
echo "Next: bash scripts/run_significance.sh — encodes TEST once per row, then runs"
echo "bootstrap CIs, the re-rank sweep, error analysis, caption stats and split audits."
