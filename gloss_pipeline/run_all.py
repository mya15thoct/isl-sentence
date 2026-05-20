"""
Run full gloss pipeline in sequence:
  1. Extract keypoints from ISL-Frames-Data images
  2. Train static MLP classifier
  3. Test inference on one sample

Usage:
  nohup python -u gloss_pipeline/run_all.py > /mnt/ngan/ISL-Sequences/logs/gloss_pipeline.log 2>&1 &
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import ISL_SEQ_DIR

# ── Step 1: Extract keypoints ─────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 1/3 — Extract keypoints from ISL-Frames-Data')
print('='*60)
from gloss_pipeline.data.extract_frames import extract
extract(
    frames_dir = Path('/mnt/ngan/ISL-Frames-Data'),
    output_dir = ISL_SEQ_DIR / 'frame_keypoints',
)

# ── Step 2: Train static classifier ──────────────────────────────────────────
print('\n' + '='*60)
print('STEP 2/3 — Train static classifier')
print('='*60)
import types, argparse
from gloss_pipeline.training.train_static import train

args = argparse.Namespace(
    kp_dir     = str(ISL_SEQ_DIR / 'frame_keypoints'),
    ckpt_dir   = str(ISL_SEQ_DIR / 'checkpoints' / 'static_classifier'),
    epochs     = 200,
    batch_size = 32,
    lr         = 1e-3,
    val_split  = 0.2,
    patience   = 20,
)
train(args)

# ── Step 3: Test inference ────────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 3/3 — Test inference')
print('='*60)
from gloss_pipeline.pipeline import run

keypoints_dir = ISL_SEQ_DIR / 'keypoints'
npy_files     = sorted(keypoints_dir.rglob('*.npy'))
if npy_files:
    test_file = npy_files[0]
    print(f'Using: {test_file}')
    result = run(str(test_file))
    print(f'\nPredicted: "{result}"')
else:
    print('No test sample found — skipping inference test')

print('\n' + '='*60)
print('DONE')
print('='*60)
