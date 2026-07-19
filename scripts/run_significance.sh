#!/usr/bin/env bash
# Review-fix analysis pipeline (Issues 3, 4, 6, 7, 8 + parameter match).
# Run AFTER the ablation checkpoints exist (scripts/run_ablation.sh).
#
# Step 1 needs the GPU once per checkpoint (encode TEST -> embedding dump);
# every later step is CPU-only over the dumps and can be re-run freely.
# Outputs land in $ROOT/eval; paste them into docs/paper_facts.md placeholders.
set -uo pipefail

ROOT=${ROOT:-/mnt/recover/ngan/ISL-Sequences}
TRAIN=$ROOT/manifests/isign_retrieval_videosplit_train.csv
VAL=$ROOT/manifests/isign_retrieval_videosplit_val.csv
TEST=$ROOT/manifests/isign_retrieval_videosplit_test.csv
CKPT=$ROOT/checkpoints
EVAL=$ROOT/eval
mkdir -p "$EVAL"

# Keep Python temp off the small root disk (/tmp ENOSPC killed a run before).
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"

# ---- 0. Environment report (Issue 9) — auto-fills the repro checklist -------
python scripts/env_report.py --checkpoint "$CKPT/abl_ema50/checkpoint_last.pt" \
  | tee "$EVAL/environment.md"

# ---- 1. Encode TEST once per system: metrics JSON + embedding dump ----------
ROWS="ema50 ema50_s43 ema50_s44 no_handaware no_handaware_s43 no_handaware_s44 \
no_redundancy minilm_text ctx_hands_only ctx_hands_pose ctx_hands_face \
frozen_text clip4clip_meanp cls_pool sem080 sem090 \
signclip_style pose_only face_free face_keep"
for d in $ROWS; do
  ckpt=$CKPT/abl_$d/checkpoint_ema.pt
  if [ ! -f "$ckpt" ]; then echo "--- skip $d (no checkpoint yet)"; continue; fi
  if [ -f "$EVAL/emb_$d.pt" ]; then echo "--- skip $d (already dumped)"; continue; fi
  echo "=== ENCODE: $d ==="
  python -m src.retrieval.evaluate_rerank \
    --manifest "$TEST" --checkpoints "$ckpt" \
    --pool-sizes 0 2000 1000 \
    --out "$EVAL/abl_$d.json" --dump-embeddings "$EVAL/emb_$d.pt"
done

# 3-seed late-fusion ensembles for BOTH systems (HARP vs uniform), so the
# ensemble comparison is also significance-tested.
if [ -f "$CKPT/abl_no_handaware_s44/checkpoint_ema.pt" ] && [ ! -f "$EVAL/emb_ens_uniform.pt" ]; then
  python -m src.retrieval.evaluate_rerank --manifest "$TEST" --pool-sizes 0 \
    --checkpoints "$CKPT/abl_ema50/checkpoint_ema.pt" "$CKPT/abl_ema50_s43/checkpoint_ema.pt" "$CKPT/abl_ema50_s44/checkpoint_ema.pt" \
    --out "$EVAL/ens_harp.json" --dump-embeddings "$EVAL/emb_ens_harp.pt"
  python -m src.retrieval.evaluate_rerank --manifest "$TEST" --pool-sizes 0 \
    --checkpoints "$CKPT/abl_no_handaware/checkpoint_ema.pt" "$CKPT/abl_no_handaware_s43/checkpoint_ema.pt" "$CKPT/abl_no_handaware_s44/checkpoint_ema.pt" \
    --out "$EVAL/ens_uniform.json" --dump-embeddings "$EVAL/emb_ens_uniform.pt"
fi

# ---- 2. Paired bootstrap 95% CIs for every headline comparison (Issue 4) ----
boot () { # boot <description> <args...>
  local desc=$1; shift
  echo "=== BOOTSTRAP: $desc ==="
  python -m src.retrieval.bootstrap_compare "$@" || echo "(skipped: missing dump)"
}
boot "hand-aware vs uniform fusion (THE decisive comparison)" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_no_handaware.pt" \
  --label-a HARP --label-b uniform-fusion --out "$EVAL/boot_handaware.json"
boot "Sinkhorn vs DSL (same model)" \
  --dump-a "$EVAL/emb_ema50.pt" --method-a sinkhorn --method-b dsl \
  --out "$EVAL/boot_sinkhorn_dsl.json"
boot "Sinkhorn vs raw cosine (same model)" \
  --dump-a "$EVAL/emb_ema50.pt" --method-a sinkhorn --method-b cosine \
  --out "$EVAL/boot_sinkhorn_cosine.json"
boot "full context vs hands+pose (does face significantly help?)" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_ctx_hands_pose.pt" \
  --out "$EVAL/boot_face.json"
boot "bge-large vs MiniLM" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_minilm_text.pt" \
  --out "$EVAL/boot_text_encoder.json"
boot "redundancy grouping on vs off" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_no_redundancy.pt" \
  --out "$EVAL/boot_redundancy.json"
boot "mean vs CLS pooling for bge" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_cls_pool.pt" \
  --out "$EVAL/boot_pooling.json"
boot "fine-tuned vs frozen text encoder" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_frozen_text.pt" \
  --out "$EVAL/boot_frozen.json"
boot "HARP vs CLIP4Clip-meanP-style dual encoder" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_clip4clip_meanp.pt" \
  --out "$EVAL/boot_clip4clip.json"
boot "HARP vs SignCLIP-style reproduction" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_signclip_style.pt" \
  --out "$EVAL/boot_signclip.json"
boot "full face vs reduced-face landmarks (face-keep)" \
  --dump-a "$EVAL/emb_ema50.pt" --dump-b "$EVAL/emb_face_keep.pt" \
  --out "$EVAL/boot_facekeep.json"
boot "3-seed ensembles: HARP vs uniform fusion" \
  --dump-a "$EVAL/emb_ens_harp.pt" --dump-b "$EVAL/emb_ens_uniform.pt" \
  --label-a HARP-ens --label-b uniform-ens --out "$EVAL/boot_ensembles.json"

# ---- 3. Sinkhorn/DSL sensitivity sweep + runtime (Issue 8) ------------------
python -m src.retrieval.rerank_sweep --dump "$EVAL/emb_ema50.pt" \
  --out "$EVAL/rerank_sweep.json" | tee "$EVAL/rerank_sweep.md"

# ---- 4. Error analysis (Issue 7) --------------------------------------------
# --manifest joins extra per-row columns by uid when present (category, quality
# metrics, ...); missing columns are skipped with a warning, so this is safe.
python -m src.retrieval.error_analysis --dump "$EVAL/emb_ema50.pt" \
  --manifest "$TEST" --join-columns category \
  --report "$EVAL/error_analysis.md" --out "$EVAL/error_analysis.json"

# ---- 4b. Inference-time robustness (Issue 5.3): missing hands / frame rate /
#          detector noise — GPU, ~6 TEST encodes, skip if already produced -----
if [ ! -f "$EVAL/robustness.md" ]; then
  python -m src.retrieval.robustness_eval \
    --manifest "$TEST" --checkpoint "$CKPT/abl_ema50/checkpoint_ema.pt" \
    --report "$EVAL/robustness.md" --out "$EVAL/robustness.json"
fi

# ---- 5. Duplicate-caption statistics (Issue 3) -------------------------------
python -m src.data.caption_group_stats \
  --manifests "$TRAIN" "$VAL" "$TEST" --labels train val test \
  --batch-size 128 --out "$EVAL/caption_groups.json" | tee "$EVAL/caption_groups.md"

# ---- 6. Split audits (Issue 6) -----------------------------------------------
# (a) re-verify the zero video_id overlap claim mechanically;
python -m src.data.audit_split_overlap \
  --manifests "$TRAIN" "$VAL" "$TEST" --labels train val test \
  --key-column video_id --expect-disjoint --out "$EVAL/audit_video_id.json"
# (b) inspect what other columns exist as signer/session proxies, then re-run
#     with --key-regex if any prefix encodes channel/session:
python -m src.data.audit_split_overlap \
  --manifests "$TRAIN" --labels train --list-columns

# ---- 7. Parameter-match table (Issue 1 / defense Q3) --------------------------
python scripts/param_count.py | tee "$EVAL/param_counts.md"

echo
echo "ALL ANALYSIS DONE -> $EVAL"
echo "Fill the [TO FILL] placeholders in docs/paper_facts.md from these outputs."
