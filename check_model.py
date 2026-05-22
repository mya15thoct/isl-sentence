"""
Quick check: what is /mnt/ngan/recognition/checkpoints/best_model trained on?
"""
import sys, json
import numpy as np
import tensorflow as tf
from pathlib import Path

ckpt_base = Path('/mnt/ngan/recognition/checkpoints')

# ── 1. List all files in each sub-dir ────────────────────────────────────────
print('=== Files in recognition/checkpoints ===')
for item in sorted(ckpt_base.iterdir()):
    if item.is_dir():
        files = sorted(item.iterdir())
        print(f'\n  [{item.name}/]')
        for f in files:
            print(f'    {f.name}  ({f.stat().st_size/1024:.0f} KB)')
    else:
        print(f'  {item.name}  ({item.stat().st_size/1024:.0f} KB)')

# ── 2. Load action mapping ────────────────────────────────────────────────────
mapping_path = ckpt_base / 'action_mapping_combined.json'
if mapping_path.exists():
    with open(mapping_path) as f:
        mapping = json.load(f)
    print(f'\n=== action_mapping_combined.json ===')
    if isinstance(mapping, dict):
        print(f'  Keys: {list(mapping.keys())[:10]}')
        for k, v in list(mapping.items())[:5]:
            print(f'  {k}: {v}')
        print(f'  Total entries: {len(mapping)}')
    elif isinstance(mapping, list):
        print(f'  Total classes: {len(mapping)}')
        print(f'  First 10: {mapping[:10]}')

# ── 3. Load best_model and print summary ─────────────────────────────────────
for model_name in ['best_model_combined', 'best_model', 'best_model_enriched']:
    model_path = ckpt_base / model_name
    if model_path.exists():
        print(f'\n=== Loading {model_name} ===')
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            model.summary(line_length=80)
            print(f'  Input shape : {model.input_shape}')
            print(f'  Output shape: {model.output_shape}')
            break
        except Exception as e:
            print(f'  Error: {e}')
