"""
Train EfficientNetB0 on ISL-Frames-Data raw images.

Two-phase fine-tuning:
  Phase 1: freeze EfficientNetB0, train classification head
  Phase 2: unfreeze top layers, fine-tune with low LR

Usage:
  python gloss_pipeline/training/train_cnn.py \
      --img_dir /mnt/ngan/ISL-Frames-Data \
      --ckpt_dir /mnt/ngan/ISL-Sequences/checkpoints/cnn_classifier
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

IMG_SIZE   = 224
BATCH_SIZE = 32


def build_model(num_classes: int, dropout: float = 0.4) -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    # EfficientNetB0 has its own preprocessing built-in (scales 0-255)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation='gelu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    name='gloss_probs')(x)

    return tf.keras.Model(inputs, outputs, name='EfficientNetISL')


def make_dataset(img_paths, labels, augment: bool = False) -> tf.data.Dataset:
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)
        return img, label

    def augment_image(img, label):
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0,
                                                       maxval=2, dtype=tf.int32))
        # Random zoom via crop + resize
        s = tf.random.uniform([], 0.85, 1.0)
        crop_size = tf.cast(IMG_SIZE * s, tf.int32)
        img = tf.image.random_crop(img, [crop_size, crop_size, 3])
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000) if augment else ds
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def load_dataset(img_dir: Path):
    class_dirs = sorted([d for d in img_dir.iterdir() if d.is_dir()])
    class2id   = {d.name: i for i, d in enumerate(class_dirs)}
    id2class   = {i: d.name for i, d in enumerate(class_dirs)}

    paths, labels = [], []
    for class_dir in class_dirs:
        cid = class2id[class_dir.name]
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img in sorted(class_dir.glob(ext)):
                paths.append(str(img))
                labels.append(cid)

    print(f'Loaded {len(paths)} images  |  {len(class2id)} classes')
    return paths, labels, class2id, id2class


def train(args):
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, class2id, id2class = load_dataset(Path(args.img_dir))

    p_train, p_val, y_train, y_val = train_test_split(
        paths, labels, test_size=args.val_split,
        stratify=labels, random_state=42)
    print(f'Train: {len(p_train)}  Val: {len(p_val)}')

    train_ds = make_dataset(p_train, y_train, augment=True)
    val_ds   = make_dataset(p_val,   y_val,   augment=False)

    model = build_model(len(class2id), dropout=args.dropout)
    model.summary()

    # ── Phase 1: train head only ──────────────────────────────────────────────
    print('\n── Phase 1: frozen base, training head ──')
    model.compile(
        optimizer = tf.keras.optimizers.Adam(args.lr),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy'],
    )
    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = args.phase1_epochs,
        callbacks       = [tf.keras.callbacks.EarlyStopping(
                               monitor='val_accuracy', patience=10,
                               restore_best_weights=True, mode='max', verbose=1)],
        verbose         = 1,
    )

    # ── Phase 2: unfreeze top layers, fine-tune ───────────────────────────────
    print('\n── Phase 2: fine-tuning top layers ──')
    base = model.get_layer('efficientnetb0')
    base.trainable = True
    # Freeze all but top 30 layers
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer = tf.keras.optimizers.Adam(args.lr * 0.1),
        loss      = 'sparse_categorical_crossentropy',
        metrics   = ['accuracy'],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=args.patience,
            restore_best_weights=True, mode='max', verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / 'best_cnn_classifier'),
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=8,
            min_lr=1e-7, mode='max', verbose=1),
        tf.keras.callbacks.CSVLogger(str(ckpt_dir / 'training_log.csv')),
    ]

    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = args.epochs,
        callbacks       = callbacks,
        verbose         = 1,
    )

    with open(ckpt_dir / 'class_mapping.json', 'w') as f:
        json.dump({'class2id': class2id,
                   'id2class': {str(k): v for k, v in id2class.items()}},
                  f, indent=2)

    val_pred = np.argmax(model.predict(val_ds, verbose=0), axis=-1)
    acc = float(np.mean(val_pred == np.array(y_val)))
    print(f'\nFinal val accuracy: {acc:.3f} ({acc*100:.1f}%)')
    print(f'Saved to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',       default='/mnt/ngan/ISL-Frames-Data')
    parser.add_argument('--ckpt_dir',      default=str(ISL_SEQ_DIR / 'checkpoints' / 'cnn_classifier'))
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--phase1_epochs', type=int,   default=30)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--val_split',     type=float, default=0.2)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--dropout',       type=float, default=0.4)
    args = parser.parse_args()
    train(args)
