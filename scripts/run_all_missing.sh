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

# /tmp lives on the small root disk, which other jobs have filled before —
# killing our DataLoader workers with ENOSPC in /tmp/pymp-*. Point all Python
# temp files (multiprocessing sockets, tempfile) at the big data disk instead.
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"

WITH_H2S=0
[ "${1:-}" = "--with-how2sign" ] && WITH_H2S=1

mkdir -p "$ROOT/logs"
MAINLOG=$ROOT/logs/run_all_missing_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$MAINLOG") 2>&1
echo "== run_all_missing started $(date)  (with-how2sign=$WITH_H2S) =="

wait_gpu () {
  # Training needs most of the A6000; wait until any OTHER (non-project) job
  # finishes. This is a coarse poll, not a lock — see run_locked() below for
  # what actually prevents two of OUR sessions from colliding.
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

# A process-scanning check (pgrep) polled every 10 min left a real gap: two
# sessions both training clip4clip_meanp concurrently, overwriting each
# other's checkpoint/log (observed 2026-07-20). flock is kernel-enforced and
# blocking (no poll gap), so it replaces that check for OUR OWN concurrency:
# only one run_ablation.sh / run_how2sign.sh across ALL sessions can hold this
# lock at a time; a second session blocks here until the first fully finishes.
LOCKFILE="$ROOT/train.lock"
run_locked () {
  echo "[$(date '+%F %H:%M')] waiting for the training lock ($LOCKFILE) ..."
  (
    flock -x 200
    echo "[$(date '+%F %H:%M')] lock acquired — running: $*"
    wait_gpu
    "$@"
  ) 200>"$LOCKFILE"
}

# ---- A. How2Sign data prep (CPU-bound; overlaps with the busy GPU) ----------
if [ "$WITH_H2S" = 1 ]; then
  if [ -f "$H2S/manifests/how2sign_test.csv" ]; then
    echo "== A. How2Sign data already prepared — skip =="
  elif bash cross_dataset/download_how2sign.sh; then
    echo "== A. How2Sign extract (MediaPipe, CPU, hours) + manifests =="
    for s in train val test; do
      python cross_dataset/extract_how2sign.py \
        --clips "$H2S/clips_$s" --out "$H2S/keypoints/$s" --delete-after
    done
    python cross_dataset/build_how2sign_manifest.py \
      --csv-dir "$H2S/text" --keypoint-root "$H2S/keypoints" --out-dir "$H2S/manifests"
  else
    echo "!! How2Sign download failed (Google Drive quota? resets within ~24 h)."
    echo "!! Skipping stages A and E this run — everything on iSign continues."
    echo "!! Later, just RE-RUN the same tmux command; it resumes what is missing."
    WITH_H2S=0
  fi
fi

# ---- B + C. iSign: train the missing ablation/baseline rows ------------------
echo "== C. ablation & baseline rows (rows with a checkpoint are skipped) =="
run_locked bash scripts/run_ablation.sh

# ---- D. full analysis pipeline (also locked: two sessions dumping the same
#         checkpoint concurrently could interleave-write the same .pt file) --
echo "== D. significance / sensitivity / error analysis / audits =="
run_locked bash scripts/run_significance.sh

# ---- E. How2Sign ladder ------------------------------------------------------
if [ "$WITH_H2S" = 1 ]; then
  if [ ! -f "$H2S/manifests/how2sign_test.csv" ]; then
    echo "!! How2Sign manifests missing — skip stage E (re-run later to resume)"
  else
    echo "== E. How2Sign training + official-pool eval + bootstrap =="
    run_locked bash cross_dataset/run_how2sign.sh
  fi
fi

echo "== ALL DONE $(date) =="
echo "Outputs: $ROOT/eval  (and $H2S/eval if How2Sign ran)."
echo "Fill the [TO FILL] placeholders in docs/paper_facts.md from these files."
