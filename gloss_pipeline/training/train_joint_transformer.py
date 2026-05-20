"""
Train Joint Transformer on ISL-Frames-Data keypoints with strong augmentation.

Augmentations applied on-the-fly each epoch:
  - Gaussian noise          → robust to sensor jitter and transition frames
  - Horizontal flip         → swap left/right joints (mirror signing)
  - Scale jitter            → signer at different distances
  - Coordinate shift        → signer at different positions
  - Joint dropout           → random joints missing (occlusion)

Usage:
  python gloss_pipeline/training/train_joint_transformer.py \
      --kp_dir /mnt/ngan/ISL-Sequences/frame_keypoints \
      --ckpt_dir /mnt/ngan/ISL-Sequences/checkpoints/joint_transformer
"""

import json
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR
sys.path.append(str(Path(__file__).parent.parent))
from models.joint_transformer import build_joint_transformer


# ── Augmentation ──────────────────────────────────────────────────────────────

# MediaPipe left↔right joint swap indices for horizontal flip
# Pose: swap left/right body joints
POSE_FLIP_MAP = {
    1:2, 2:1, 3:4, 4:3, 5:6, 6:5, 7:8, 8:7,
    9:10, 10:9, 11:12, 12:11, 13:14, 14:13, 15:16, 16:15,
    17:18, 18:17, 19:20, 20:19, 21:22, 22:21,
    23:24, 24:23, 25:26, 26:25, 27:28, 28:27, 29:30, 30:29, 31:32, 32:31,
}


def augment(kp: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply random augmentations to a single keypoint (1662,)."""
    kp = kp.copy()

    # Gaussian noise
    if np.random.random() < p:
        kp += np.random.normal(0, 0.01, kp.shape).astype(np.float32)

    # Scale jitter
    if np.random.random() < p:
        scale = np.random.uniform(0.85, 1.15)
        kp *= scale

    # Coordinate shift
    if np.random.random() < p:
        shift = np.random.uniform(-0.05, 0.05)
        kp += shift

    # Joint dropout — zero out random joints
    if np.random.random() < p * 0.5:
        pose = kp[:132].reshape(33, 4)
        drop_idx = np.random.choice(33, size=np.random.randint(1, 5), replace=False)
        pose[drop_idx] = 0.0
        kp[:132] = pose.flatten()

    # Horizontal flip — swap left/right
    if np.random.random() < p * 0.5:
        pose = kp[:132].reshape(33, 4).copy()
        for l, r in POSE_FLIP_MAP.items():
            pose[l], pose[r] = pose[r].copy(), pose[l].copy()
        pose[:, 0] = 1.0 - pose[:, 0]   # mirror x
        kp[:132] = pose.flatten()
        # Swap left and right hand
        lh = kp[1536:1599].copy()
        rh = kp[1599:].copy()
        lh[0::3] = 1.0 - lh[0::3]
        rh[0::3] = 1.0 - rh[0::3]
        kp[1536:1599] = rh
        kp[1599:]     = lh

    return kp


class AugmentedDataset(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, augment_factor=5, training=True):
        self.X            = X
        self.y            = y
        self.batch_size   = batch_size
        self.augment_factor = augment_factor
        self.training     = training

    def __len__(self):
        n = len(self.X) * (self.augment_factor if self.training else 1)
        return (n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = (idx * self.batch_size) % len(self.X)
        end   = min(start + self.batch_size, len(self.X))
        X_batch = self.X[start:end].copy()
        y_batch = self.y[start:end]
        if self.training:
            X_batch = np.stack([augment(x) for x in X_batch])
        return X_batch, y_batch


def load_dataset(kp_dir: Path):
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
            X.append(kp)
            y.append(cid)

    X = np.stack(X)
    y = np.array(y, dtype=np.int32)
    print(f'Loaded {len(X)} samples  |  {len(class2id)} classes')
    return X, y, class2id, id2class


def train(args):
    kp_dir   = Path(args.kp_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X, y, class2id, id2class = load_dataset(kp_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, stratify=y, random_state=42)
    print(f'Train: {len(X_train)}  Val: {len(X_val)}')

    train_gen = AugmentedDataset(X_train, y_train, args.batch_size,
                                  augment_factor=args.augment_factor, training=True)
    val_gen   = AugmentedDataset(X_val, y_val, args.batch_size,
                                  augment_factor=1, training=False)

    model = build_joint_transformer(
        num_classes = len(class2id),
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        ff_dim      = args.ff_dim,
        num_layers  = args.num_layers,
        dropout     = args.dropout,
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
            filepath=str(ckpt_dir / 'best_joint_transformer'),
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=10,
            min_lr=1e-6, mode='max', verbose=1),
        tf.keras.callbacks.CSVLogger(str(ckpt_dir / 'training_log.csv')),
    ]

    model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = args.epochs,
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
    parser.add_argument('--kp_dir',          default=str(ISL_SEQ_DIR / 'frame_keypoints'))
    parser.add_argument('--ckpt_dir',         default=str(ISL_SEQ_DIR / 'checkpoints' / 'joint_transformer'))
    parser.add_argument('--epochs',           type=int,   default=300)
    parser.add_argument('--batch_size',       type=int,   default=64)
    parser.add_argument('--lr',               type=float, default=3e-4)
    parser.add_argument('--val_split',        type=float, default=0.2)
    parser.add_argument('--patience',         type=int,   default=30)
    parser.add_argument('--augment_factor',   type=int,   default=10)
    parser.add_argument('--d_model',          type=int,   default=128)
    parser.add_argument('--num_heads',        type=int,   default=4)
    parser.add_argument('--ff_dim',           type=int,   default=256)
    parser.add_argument('--num_layers',       type=int,   default=4)
    parser.add_argument('--dropout',          type=float, default=0.3)
    args = parser.parse_args()
    train(args)
