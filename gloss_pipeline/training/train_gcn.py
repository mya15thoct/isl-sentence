"""
Train GCN classifier on ISL-Frames-Data pseudo-sequences.

Each image → keypoint (1662,) → repeat seq_len times → pseudo-sequence (seq_len, 1662)
The word model BiLSTM extracts motion dynamic features (frozen).
GCN + head trains to classify the sign.

Usage:
  python gloss_pipeline/training/train_gcn.py \
      --kp_dir /mnt/ngan/ISL-Sequences/frame_keypoints \
      --ckpt_dir /mnt/ngan/ISL-Sequences/checkpoints/gcn_classifier
"""

import json
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR, VIDEO_WORD_MODEL_PATH
sys.path.append(str(Path(__file__).parent.parent))
from models.gcn_classifier import build_gcn_classifier


def load_dataset(kp_dir: Path, seq_len: int):
    """Load frame keypoints and create pseudo-sequences by repeating each frame."""
    class_dirs = sorted([d for d in kp_dir.iterdir() if d.is_dir()])
    class2id   = {d.name: i for i, d in enumerate(class_dirs)}
    id2class   = {i: d.name for i, d in enumerate(class_dirs)}

    X, y = [], []
    for class_dir in class_dirs:
        cid = class2id[class_dir.name]
        for npy in sorted(class_dir.glob('*.npy')):
            kp = np.load(npy).astype(np.float32)
            if kp.shape[0] != 1662:
                continue
            # Pseudo-sequence: repeat single frame seq_len times
            seq = np.tile(kp, (seq_len, 1))   # (seq_len, 1662)
            X.append(seq)
            y.append(cid)

    X = np.stack(X)
    y = np.array(y, dtype=np.int32)
    print(f'Loaded {len(X)} samples  |  {len(class2id)} classes  |  seq shape: {X.shape}')
    return X, y, class2id, id2class


def train(args):
    kp_dir   = Path(args.kp_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X, y, class2id, id2class = load_dataset(kp_dir, args.seq_len)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=42)
    print(f'Train: {len(X_train)}  Val: {len(X_val)}')

    model = build_gcn_classifier(
        num_classes      = len(class2id),
        word_model_path  = args.word_model,
        seq_len          = args.seq_len,
        freeze_extractor = not args.unfreeze,
    )
    model.summary()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(args.lr),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy'],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=args.patience,
            restore_best_weights=True, mode='max', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / 'best_gcn_classifier'),
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=10,
            min_lr=1e-6, mode='max', verbose=1),
        tf.keras.callbacks.CSVLogger(str(ckpt_dir / 'training_log.csv')),
    ]

    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        callbacks       = callbacks,
        verbose         = 1,
    )

    with open(ckpt_dir / 'class_mapping.json', 'w') as f:
        json.dump({'class2id': class2id,
                   'id2class': {str(k): v for k, v in id2class.items()}},
                  f, indent=2)

    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=-1)
    acc = float(np.mean(val_pred == y_val))
    print(f'\nFinal val accuracy: {acc:.3f} ({acc*100:.1f}%)')
    print(f'Saved to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kp_dir',    default=str(ISL_SEQ_DIR / 'frame_keypoints'))
    parser.add_argument('--ckpt_dir',  default=str(ISL_SEQ_DIR / 'checkpoints' / 'gcn_classifier'))
    parser.add_argument('--word_model',default=str(VIDEO_WORD_MODEL_PATH))
    parser.add_argument('--seq_len',   type=int,   default=30)
    parser.add_argument('--epochs',    type=int,   default=200)
    parser.add_argument('--batch_size',type=int,   default=32)
    parser.add_argument('--lr',        type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--patience',  type=int,   default=20)
    parser.add_argument('--unfreeze',  action='store_true',
                        help='Unfreeze word model feature extractor')
    args = parser.parse_args()
    train(args)
