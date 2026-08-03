#!/usr/bin/env bash
# Cross-lingual study: train ONE pose-text model on ISL (iSign) + ASL (How2Sign)
# jointly (shared English text space), then evaluate on EACH language's test set.
# Compares hand-aware (HARP) vs uniform fusion to ask: does the hand-centric design
# transfer across sign languages better than holistic fusion?
#
# Same recipe as the iSign headline (no per-dataset tuning). Run after the joint
# manifests exist. Baselines to compare against (already trained, monolingual):
#   iSign    HARP 0.564 / uniform 0.518   (test full pool, Sinkhorn R@10)
#   How2Sign HARP 0.201 / uniform 0.153
set -uo pipefail

R=${R:-/mnt/recover/ngan/ISL-Sequences}
H=${H:-/mnt/recover/ngan/How2Sign}
J=$R/manifests            # joint manifests land here
CKPT=$R/checkpoints
LOG=$R/logs
BGE="BAAI/bge-large-en-v1.5"
export TMPDIR=$R/tmp; mkdir -p "$TMPDIR" "$LOG"

# ---- 1. Build joint train/val manifests (ISL + ASL) --------------------------
if [ ! -f "$J/joint_train.csv" ]; then
  python cross_dataset/build_joint_manifest.py \
    --manifests "$R/manifests/isign_retrieval_videosplit_train.csv" "$H/manifests/how2sign_train.csv" \
    --labels isign how2sign --out "$J/joint_train.csv"
fi
if [ ! -f "$J/joint_val.csv" ]; then
  python cross_dataset/build_joint_manifest.py \
    --manifests "$R/manifests/isign_retrieval_videosplit_val.csv" "$H/manifests/how2sign_val.csv" \
    --labels isign how2sign --out "$J/joint_val.csv"
fi

COMMON="--manifest $J/joint_train.csv --val-manifest $J/joint_val.csv --text-column canonical_text \
  --epochs 50 --batch-size 128 --num-workers 4 --max-frames 512 \
  --sample-mode random --augment --augment-prob 0.8 \
  --lr 2e-4 --text-lr 1e-5 --warmup-epochs 2 --density-weight 0 \
  --ema-decay 0.999 --seed 42 --device cuda --amp --print-every 100"

run () {  # run <name> <extra-flags...>
  local name=$1; shift
  if [ -f "$CKPT/joint_$name/checkpoint_ema.pt" ]; then echo "=== SKIP: joint_$name ==="; return 0; fi
  echo "=== RUN: joint_$name ==="
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u src/retrieval/train.py $COMMON \
       --save-dir "$CKPT/joint_$name" "$@" > "$LOG/joint_$name.log" 2>&1; then
    echo "=== DONE: joint_$name ==="
  else
    echo "=== FAILED: joint_$name (see $LOG/joint_$name.log) ==="
  fi
}

# ---- 2. Train joint HARP + joint uniform-fusion ------------------------------
run harp         --text-model "$BGE" --embedding-dim 1024 --hand-aware
run no_handaware --text-model "$BGE" --embedding-dim 1024

# ---- 3. Evaluate each joint model on BOTH languages' test sets ---------------
echo "=== EVAL joint models on each language ==="
for m in harp no_handaware; do
  ck=$CKPT/joint_$m/checkpoint_ema.pt
  [ -f "$ck" ] || { echo "skip eval $m (no ckpt)"; continue; }
  python -m src.retrieval.evaluate_rerank --manifest "$R/manifests/isign_retrieval_videosplit_test.csv" \
    --checkpoints "$ck" --pool-sizes 0 --out "$R/eval/joint_${m}_on_isign.json"
  python -m src.retrieval.evaluate_rerank --manifest "$H/manifests/how2sign_test.csv" \
    --checkpoints "$ck" --pool-sizes 0 --out "$H/eval/joint_${m}_on_how2sign.json"
done

echo
echo "=== COMPARE (full pool, Sinkhorn R@10) ==="
python3 - <<'PY'
import json, os
R="/mnt/recover/ngan/ISL-Sequences/eval"; H="/mnt/recover/ngan/How2Sign/eval"
def r10(p):
    try:
        d=json.load(open(p)); k=[x for x in d if 'full' in x][0]
        return d[k]['sinkhorn']['v2t']['exact']['r10']
    except Exception: return None
rows=[("iSign  HARP  mono", 0.564), ("iSign  unif  mono", 0.518),
      ("How2S  HARP  mono", 0.201), ("How2S  unif  mono", 0.153),
      ("iSign  HARP  joint", r10(f"{R}/joint_harp_on_isign.json")),
      ("iSign  unif  joint", r10(f"{R}/joint_no_handaware_on_isign.json")),
      ("How2S  HARP  joint", r10(f"{H}/joint_harp_on_how2sign.json")),
      ("How2S  unif  joint", r10(f"{H}/joint_no_handaware_on_how2sign.json"))]
for name,v in rows:
    print(f"  {name:20s} {'' if v is None else f'{v:.3f}'}")
print("\nRead: does JOINT beat MONO on each language? does HARP gain MORE from joint than uniform?")
PY
echo "Done. If joint HARP > mono AND gains more than uniform -> hand-aware transfers better across languages."
