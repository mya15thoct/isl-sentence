#!/usr/bin/env bash
# ONE-SHOT runner: produces every result still missing for the Q1 revision.
# Safe to re-run after any interruption — every stage skips work already done.
#
# Start it detached in tmux on the server:
#   tmux new -d -s allrun 'bash scripts/run_all_missing.sh --with-how2sign'
# Watch progress:
#   tail -f /mnt/recover/ngan/ISL-Sequences/logs/run_all_missing_*.log
#
# Stages:
#   A. How2Sign download + MediaPipe extraction + manifests (CPU-only — runs
#      first, while the GPU is still busy)          [only with --with-how2sign]
#   B. wait until the GPU is free (another job may occupy it; checks every 10 min)
#   C. scripts/run_ablation.sh       — trains ONLY the missing ablation rows
#   D. scripts/run_significance.sh   — dumps, bootstrap CIs, sweep, error analysis,
#                                      caption stats, split audits, param table
#   E. cross_dataset/run_how2sign.sh — How2Sign ladder + eval + bootstrap  [flag]
set -uo pipefail

ROOT=${ROOT:-/mnt/recover/ngan/ISL-Sequences}
H2S=${H2S:-/mnt/recover/ngan/How2Sign}
WITH_H2S=0
[ "${1:-}" = "--with-how2sign" ] && WITH_H2S=1

mkdir -p "$ROOT/logs"
MAINLOG=$ROOT/logs/run_all_missing_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$MAINLOG") 2>&1
echo "== run_all_missing started $(date)  (with-how2sign=$WITH_H2S) =="

wait_gpu () {
  # Training needs most of the A6000; wait until the current occupant finishes.
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-0}" -lt 8000 ]; then
      echo "GPU free enough (${used} MiB used) — proceeding"
      break
    fi
    echo "[$(date '+%F %H:%M')] GPU busy (${used} MiB used) — retry in 10 min"
    sleep 600
  done
}

# ---- A. How2Sign data prep (CPU-bound; overlaps with the busy GPU) ----------
if [ "$WITH_H2S" = 1 ]; then
  if [ -f "$H2S/manifests/how2sign_test.csv" ]; then
    echo "== A. How2Sign data already prepared — skip =="
  else
    echo "== A. How2Sign download + extract + manifests (hours, CPU) =="
    bash cross_dataset/download_how2sign.sh
    python cross_dataset/extract_how2sign.py --delete-after
    python cross_dataset/build_how2sign_manifest.py
  fi
fi

# ---- B + C. iSign: train the missing ablation/baseline rows ------------------
wait_gpu
echo "== C. ablation & baseline rows (rows with a checkpoint are skipped) =="
bash scripts/run_ablation.sh

# ---- D. full analysis pipeline ----------------------------------------------
echo "== D. significance / sensitivity / error analysis / audits =="
bash scripts/run_significance.sh

# ---- E. How2Sign ladder ------------------------------------------------------
if [ "$WITH_H2S" = 1 ]; then
  wait_gpu
  echo "== E. How2Sign training + official-pool eval + bootstrap =="
  bash cross_dataset/run_how2sign.sh
fi

echo "== ALL DONE $(date) =="
echo "Outputs: $ROOT/eval  (and $H2S/eval if How2Sign ran)."
echo "Fill the [TO FILL] placeholders in docs/paper_facts.md from these files."
