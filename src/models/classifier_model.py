"""
Sentence Classifier for ISL Recognition (Approach D).

Instead of sequence generation (CTC / Seq2Seq), reframes the problem
as 99-class classification: encoder → GlobalAveragePooling → Dense(99).

Why this works with small data:
  - No alignment learning needed (unlike CTC)
  - Cross-entropy loss is simple and stable
  - Transfer from word model gives strong feature initialisation
  - 486 samples × 10 augmentation / 99 classes ≈ 49 samples/class

Architecture:
  Input (B, T, 1662)
    → MLP branches (pose/face/hand)  [same as sentence_model.py — transfer-compatible]
    → shared Dense layers
    → BiLSTM × 2  (return_sequences=True)
    → GlobalAveragePooling1D          ← collapse temporal (B, T, 64) → (B, 64)
    → Dense(128, relu) + Dropout(0.4)
    → Dense(num_sentences, softmax)   ← 99-class output

Transfer:
  Layer names match sentence_model.py / hybrid.py exactly so the same
  name-based weight copy works for BiLSTM + MLP branches.

Usage:
  model = build_classifier_model(num_sentences=99, word_model_path='...')
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHECKPOINT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# MLP BRANCHES  (identical names to sentence_model.py for weight transfer)
# ─────────────────────────────────────────────────────────────────────────────

def _pose_branch(input_dim: int, name: str):
    return tf.keras.Sequential([
        layers.Dense(128, activation='relu', name=f'{name}_d1'),
        layers.BatchNormalization(name=f'{name}_bn1'),
        layers.Dense(64,  activation='relu', name=f'{name}_d2'),
        layers.BatchNormalization(name=f'{name}_bn2'),
    ], name=name)


def _face_branch(input_dim: int, name: str):
    return tf.keras.Sequential([
        layers.Dense(512, activation='relu', name=f'{name}_d1'),
        layers.BatchNormalization(name=f'{name}_bn1'),
        layers.Dropout(0.3, name=f'{name}_drop1'),
        layers.Dense(256, activation='relu', name=f'{name}_d2'),
        layers.BatchNormalization(name=f'{name}_bn2'),
        layers.Dense(128, activation='relu', name=f'{name}_d3'),
        layers.BatchNormalization(name=f'{name}_bn3'),
        layers.Dense(128, activation='relu', name=f'{name}_d4'),
        layers.BatchNormalization(name=f'{name}_bn4'),
    ], name=name)


def _hand_branch(input_dim: int, name: str):
    return tf.keras.Sequential([
        layers.Dense(256, activation='relu', name=f'{name}_d1'),
        layers.BatchNormalization(name=f'{name}_bn1'),
        layers.Dense(128, activation='relu', name=f'{name}_d2'),
        layers.BatchNormalization(name=f'{name}_bn2'),
        layers.Dense(64,  activation='relu', name=f'{name}_d3'),
        layers.BatchNormalization(name=f'{name}_bn3'),
    ], name=name)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier_model(
    num_sentences:   int,
    word_model_path: str,
    freeze_encoder:  bool = False,
) -> Model:
    """
    Build sentence classifier and transfer encoder weights from word model.

    Args:
        num_sentences:   Number of sentence classes (e.g. 99)
        word_model_path: Path to best_model_combined (SavedModel)
        freeze_encoder:  Freeze all layers except classifier head.
                         Useful for two-phase training: warm up head first,
                         then unfreeze for full fine-tuning.

    Returns:
        Compiled-ready Keras Model
    """
    # ── Encoder (identical structure to sentence_model.py) ────────────────────
    inputs = layers.Input(shape=(None, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_kp = layers.Lambda(lambda t: t[:, :, :132],     name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],    name='hand_split')(x)

    pose_branch = _pose_branch(132,  'pose')
    face_branch = _face_branch(1404, 'face')
    hand_branch = _hand_branch(126,  'hand')

    pose_feat = layers.TimeDistributed(pose_branch, name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(face_branch, name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(hand_branch, name='hand_features')(hand_kp)

    merged = layers.Concatenate(name='feature_fusion')([pose_feat, face_feat, hand_feat])

    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1')(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name='bilstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True), name='bilstm2')(x)  # (B, T, 64)
    x = layers.Dropout(0.3)(x)

    # ── Classification head ───────────────────────────────────────────────────
    # Masked mean pooling: compute average only over valid (non-padded) frames.
    # Padded frames have all-zero keypoints; their mask value = 0 so they are
    # excluded from both the sum and the count.
    # Using `inputs` (before MLP) to derive the mask avoids bias contamination.
    frame_mask = layers.Lambda(
        lambda t: tf.cast(
            tf.reduce_any(tf.not_equal(t, 0.0), axis=-1, keepdims=True),
            tf.float32,
        ),
        name='frame_mask',
    )(inputs)  # (B, T, 1)  — 1 valid, 0 padded

    x = layers.Lambda(
        lambda args: (
            tf.reduce_sum(args[0] * args[1], axis=1) /
            tf.maximum(tf.reduce_sum(args[1], axis=1), 1.0)
        ),
        name='masked_pool',
    )([x, frame_mask])  # (B, 64)
    x = layers.Dense(128, activation='relu', name='cls_dense')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(
        num_sentences, activation='softmax', name='sentence_probs')(x)

    model = Model(inputs=inputs, outputs=outputs, name='SentenceClassifier')

    # ── Transfer weights from word model ──────────────────────────────────────
    print(f'[Classifier] Loading word model from: {word_model_path}')
    try:
        word_model = tf.keras.models.load_model(str(word_model_path))
    except Exception as e:
        print(f'[WARNING] Could not load word model: {e}')
        print('          Proceeding with random initialisation.')
        return model

    transferred, skipped = 0, 0
    transferred_names = []
    for layer in model.layers:
        try:
            src = word_model.get_layer(layer.name)
            w = src.get_weights()
            if w:
                layer.set_weights(w)
                transferred += 1
                transferred_names.append(layer.name)
        except Exception:
            skipped += 1

    print(f'[Classifier] Weights transferred: {transferred}  |  Skipped (new): {skipped}')
    print(f'[Classifier] Transferred layers : {transferred_names}')

    if freeze_encoder:
        head_layers = {'temporal_pool', 'cls_dense', 'sentence_probs'}
        for layer in model.layers:
            if layer.name not in head_layers:
                layer.trainable = False
        print('[Classifier] Encoder frozen — only classification head will train')

    return model


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np

    NUM_SENTENCES = 99
    WORD_MODEL = '/mnt/ngan/ISL-Sequences/checkpoints/best_model_combined'

    model = build_classifier_model(NUM_SENTENCES, WORD_MODEL)
    model.summary()

    x = np.random.rand(4, 150, 1662).astype(np.float32)
    out = model(x, training=False)
    print(f'\nInput  shape : {x.shape}')
    print(f'Output shape : {out.shape}')   # (4, 99)
    assert out.shape == (4, NUM_SENTENCES)
    print('[OK] Output shape correct')
    print(f'[OK] Probabilities sum to 1: {out.numpy().sum(axis=-1)}')
