"""
Data inventory: count all frame keypoints and sentence videos.
Run from repo root:
  python check_data.py
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent))
from config import ISL_SEQ_DIR

# ── 1. Frame keypoints ────────────────────────────────────────────────────────
kp_dir = ISL_SEQ_DIR / 'frame_keypoints'
print('=' * 60)
print('FRAME KEYPOINTS (word classifier data)')
print('=' * 60)

if kp_dir.exists():
    class_dirs = sorted([d for d in kp_dir.iterdir() if d.is_dir()])
    counts = {}
    total_frames = 0
    for cd in class_dirs:
        n = len(list(cd.glob('*.npy')))
        counts[cd.name] = n
        total_frames += n

    print(f'  Classes (words)  : {len(class_dirs)}')
    print(f'  Total .npy files : {total_frames}')
    if counts:
        vals = list(counts.values())
        print(f'  Avg / class      : {np.mean(vals):.1f}')
        print(f'  Min / class      : {np.min(vals)}')
        print(f'  Max / class      : {np.max(vals)}')
        print(f'  Median / class   : {np.median(vals):.1f}')
        print()
        print('  Per-class breakdown:')
        for name, n in sorted(counts.items(), key=lambda x: -x[1]):
            bar = '#' * n
            print(f'    {name:<30s} {n:4d}  {bar}')
else:
    print(f'  [NOT FOUND] {kp_dir}')

# ── 2. Sentence videos ────────────────────────────────────────────────────────
print()
print('=' * 60)
print('SENTENCE VIDEOS')
print('=' * 60)

video_dir = ISL_SEQ_DIR / 'videos'
if video_dir.exists():
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    all_videos = [f for f in video_dir.rglob('*') if f.suffix.lower() in exts]
    subdirs    = sorted([d for d in video_dir.iterdir() if d.is_dir()])

    if subdirs:
        sent_counts = {}
        for sd in subdirs:
            vids = [f for f in sd.iterdir() if f.suffix.lower() in exts]
            sent_counts[sd.name] = len(vids)
        vals = list(sent_counts.values())
        print(f'  Sentences        : {len(sent_counts)}')
        print(f'  Total videos     : {sum(vals)}')
        print(f'  Avg / sentence   : {np.mean(vals):.1f}')
        print(f'  Min / sentence   : {np.min(vals)}')
        print(f'  Max / sentence   : {np.max(vals)}')
        print(f'  Median / sentence: {np.median(vals):.1f}')
        print()
        print('  Per-sentence breakdown:')
        for name, n in sorted(sent_counts.items(), key=lambda x: -x[1]):
            bar = '#' * n
            print(f'    {name:<40s} {n:3d}  {bar}')
    else:
        print(f'  Total videos (flat): {len(all_videos)}')
else:
    print(f'  [NOT FOUND] {video_dir}')

# ── 3. Sentence keypoints ─────────────────────────────────────────────────────
print()
print('=' * 60)
print('SENTENCE KEYPOINTS (extracted .npy sequences)')
print('=' * 60)

seq_dir = ISL_SEQ_DIR / 'keypoints'
if seq_dir.exists():
    subdirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
    if subdirs:
        sent_counts = {}
        for sd in subdirs:
            n = len(list(sd.glob('*.npy')))
            sent_counts[sd.name] = n
        vals = list(sent_counts.values())
        print(f'  Sentences        : {len(sent_counts)}')
        print(f'  Total sequences  : {sum(vals)}')
        print(f'  Avg / sentence   : {np.mean(vals):.1f}')
        print(f'  Min / sentence   : {np.min(vals)}')
        print(f'  Max / sentence   : {np.max(vals)}')
    else:
        npys = list(seq_dir.glob('*.npy'))
        print(f'  Total sequences (flat): {len(npys)}')
else:
    print(f'  [NOT FOUND] {seq_dir}')

# ── 4. Labels ─────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('LABELS CSV')
print('=' * 60)

label_path = ISL_SEQ_DIR / 'labels.csv'
if label_path.exists():
    lines = label_path.read_text().strip().splitlines()
    print(f'  Rows (incl. header): {len(lines)}')
    print(f'  First 5 rows:')
    for l in lines[:5]:
        print(f'    {l}')
else:
    print(f'  [NOT FOUND] {label_path}')

print()
print('Done.')
