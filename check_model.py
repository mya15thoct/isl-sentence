"""
Check accuracy of recognition/checkpoints models on recognition/sequences data.
"""
import sys, json
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import SERVER_BASE

ckpt_base = Path('/mnt/ngan/recognition/checkpoints')
seq_dir   = SERVER_BASE / 'recognition' / 'sequences'

# ── 1. Load action mapping ────────────────────────────────────────────────────
mapping_path = ckpt_base / 'action_mapping_combined.json'
if not mapping_path.exists():
    print(f'[NOT FOUND] {mapping_path}')
    sys.exit(1)

with open(mapping_path) as f:
    raw = json.load(f)

# mapping may be {class_name: id} or {id: class_name}
if isinstance(raw, dict):
    # detect direction
    first_key = next(iter(raw))
    if first_key.isdigit():
        id2class = {int(k): v for k, v in raw.items()}
        class2id = {v: k for k, v in id2class.items()}
    else:
        class2id = raw
        id2class = {v: k for k, v in raw.items()}
elif isinstance(raw, list):
    class2id = {c: i for i, c in enumerate(raw)}
    id2class = {i: c for i, c in enumerate(raw)}

print(f'Mapping: {len(class2id)} classes')
print(f'Sample classes: {list(class2id.keys())[:10]}')

# ── 2. Load data ──────────────────────────────────────────────────────────────
print(f'\nLoading sequences from {seq_dir} ...')
SEQ_LEN = 93  # model expects this fixed length

def pad_or_trim(seq, length):
    T = seq.shape[0]
    if T >= length:
        return seq[:length]
    pad = np.zeros((length - T, seq.shape[1]), dtype=np.float32)
    return np.concatenate([seq, pad], axis=0)

X, y = [], []
missing = []
for class_dir in sorted(seq_dir.iterdir()):
    if not class_dir.is_dir():
        continue
    cname = class_dir.name
    if cname not in class2id:
        missing.append(cname)
        continue
    cid = class2id[cname]
    for npy in sorted(class_dir.glob('*.npy')):
        seq = np.load(npy).astype(np.float32)
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)
        seq = pad_or_trim(seq, SEQ_LEN)   # (93, 1662)
        X.append(seq)
        y.append(cid)

if missing:
    print(f'  Classes not in mapping ({len(missing)}): {missing[:5]}...')

X = np.stack(X).astype(np.float32)   # (N, 93, 1662)
y = np.array(y, dtype=np.int32)
print(f'  Loaded: {len(X)} samples, {len(set(y))} classes')
print(f'  Shape : {X.shape}')

# ── 3. Evaluate each checkpoint ──────────────────────────────────────────────
for model_name in ['best_model']:
    model_path = ckpt_base / model_name
    if not model_path.exists():
        print(f'\n[SKIP] {model_name} not found')
        continue
    print(f'\n=== {model_name} ===')
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f'  Input : {model.input_shape}  Output: {model.output_shape}')

        exp = model.input_shape
        print(f'  Model expects: {exp}  |  Data shape: {X.shape}')

        preds = np.argmax(model.predict(X, batch_size=128, verbose=0), axis=1)
        acc   = float(np.mean(preds == y))
        print(f'  Accuracy on recognition/sequences: {acc:.4f} ({acc*100:.1f}%)')
    except Exception as e:
        print(f'  Error: {e}')
