"""
Diagnose why the word model isn't working with sliding window input.
Checks: input shape, seq_len, masking, and tests on known isolated samples.

Usage:
  python src/inference/diagnose_word_model.py
"""
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    WORD_MODEL_PATH, ACTION_MAPPING_PATH,
    ISL_SEQ_DIR, WORD_SEQUENCE_PATH,
)

def diagnose():
    # ── 1. Load word model and inspect ────────────────────────────────────────
    print('=' * 60)
    print('WORD MODEL DIAGNOSIS')
    print('=' * 60)

    model_path = str(WORD_MODEL_PATH)
    print(f'\n[1] Loading model from: {model_path}')
    model = tf.keras.models.load_model(model_path)

    print(f'\n[2] Model input shape : {model.input_shape}')
    print(f'    Model output shape: {model.output_shape}')

    seq_len = model.input_shape[1]
    print(f'    seq_len = {seq_len}  (None = variable)')

    # ── 2. Check for Masking layer ────────────────────────────────────────────
    has_masking = False
    print(f'\n[3] Layer summary:')
    for layer in model.layers[:10]:
        print(f'    {layer.name:30s}  {layer.__class__.__name__:20s}  output={layer.output_shape}')
        if 'masking' in layer.__class__.__name__.lower():
            has_masking = True
    if len(model.layers) > 10:
        print(f'    ... ({len(model.layers)} total layers)')
    print(f'    Has Masking layer: {has_masking}')

    # ── 3. Load action mapping ────────────────────────────────────────────────
    with open(ACTION_MAPPING_PATH, encoding='utf-8') as f:
        mapping = json.load(f)
    idx2gloss = {int(k): v for k, v in mapping.items()}
    print(f'\n[4] Vocab: {len(idx2gloss)} classes')

    # ── 4. Test with random input at different seq_lens ───────────────────────
    print(f'\n[5] Testing with random input at different seq_lens:')
    for test_len in [30, 60, 100, 200]:
        try:
            x = np.random.rand(1, test_len, 1662).astype(np.float32) * 0.5
            pred = model.predict(x, verbose=0)
            top_idx = int(np.argmax(pred[0]))
            top_conf = float(pred[0][top_idx])
            top3_idx = np.argsort(pred[0])[-3:][::-1]
            top3 = [(idx2gloss.get(int(i), '?'), f'{pred[0][i]:.4f}') for i in top3_idx]
            print(f'    seq_len={test_len:3d}  →  top={idx2gloss.get(top_idx,"?")} ({top_conf:.4f})  top3={top3}')
        except Exception as e:
            print(f'    seq_len={test_len:3d}  →  ERROR: {e}')

    # ── 5. Test on known isolated word samples ────────────────────────────────
    print(f'\n[6] Testing on ISOLATED word samples (from word model training data):')

    # Try to find word model training sequences
    word_seq_dir = WORD_SEQUENCE_PATH
    if not word_seq_dir.exists():
        # Fallback: look in common locations
        for alt in [
            Path('/mnt/ngan/recognition/sequences'),
            Path('/mnt/ngan/ISL-Sequences/word'),
            ISL_SEQ_DIR / 'word',
        ]:
            if alt.exists():
                word_seq_dir = alt
                break

    if word_seq_dir.exists():
        print(f'    Word sequences dir: {word_seq_dir}')
        class_dirs = sorted([d for d in word_seq_dir.iterdir() if d.is_dir()])[:5]

        for class_dir in class_dirs:
            npy_files = sorted(class_dir.glob('*.npy'))[:1]
            for npy in npy_files:
                arr = np.load(npy)
                T = arr.shape[0]

                # Try at original length (padded to seq_len if needed)
                if seq_len is not None:
                    if T < seq_len:
                        padded = np.zeros((seq_len, 1662), dtype=np.float32)
                        padded[:T] = arr
                        x = padded[np.newaxis]
                    else:
                        x = arr[:seq_len][np.newaxis]
                else:
                    x = arr[np.newaxis]

                pred = model.predict(x.astype(np.float32), verbose=0)
                top_idx = int(np.argmax(pred[0]))
                top_conf = float(pred[0][top_idx])
                top3_idx = np.argsort(pred[0])[-3:][::-1]
                top3 = [(idx2gloss.get(int(i), '?'), f'{pred[0][i]:.4f}') for i in top3_idx]
                print(f'    {class_dir.name:20s} (T={T:3d}) → top={idx2gloss.get(top_idx,"?")} '
                      f'({top_conf:.4f})  top3={top3}')
    else:
        print(f'    Word sequences not found at {WORD_SEQUENCE_PATH}')
        print(f'    Skipping isolated word test')

    # ── 6. Test on sentence sequence with different window sizes ──────────────
    print(f'\n[7] Testing on SENTENCE samples with different window sizes:')

    sent_seq_dir = ISL_SEQ_DIR / 'sequences'
    if sent_seq_dir.exists():
        # Find first available sentence
        for sent_dir in sorted(sent_seq_dir.iterdir()):
            if not sent_dir.is_dir():
                continue
            npy_files = sorted(sent_dir.glob('*.npy'))
            if not npy_files:
                continue

            arr = np.load(npy_files[0])
            T = arr.shape[0]
            print(f'    Sentence: {sent_dir.name}  T={T}  file={npy_files[0].name}')

            for win_size in [30, 60, T]:
                window = arr[:min(win_size, T)]
                W = window.shape[0]

                if seq_len is not None:
                    if W < seq_len:
                        padded = np.zeros((seq_len, 1662), dtype=np.float32)
                        padded[:W] = window
                        x = padded[np.newaxis]
                    else:
                        x = window[:seq_len][np.newaxis]
                else:
                    x = window[np.newaxis]

                pred = model.predict(x.astype(np.float32), verbose=0)
                top_idx = int(np.argmax(pred[0]))
                top_conf = float(pred[0][top_idx])
                top3_idx = np.argsort(pred[0])[-3:][::-1]
                top3 = [(idx2gloss.get(int(i), '?'), f'{pred[0][i]:.4f}') for i in top3_idx]
                print(f'      W={W:3d} (pad→{seq_len or W}) → top={idx2gloss.get(top_idx,"?")} '
                      f'({top_conf:.4f})  top3={top3}')
            break  # just first sentence
    else:
        print(f'    Sentence sequences not found')

    print(f'\n{"=" * 60}')
    print('DIAGNOSIS COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    diagnose()
