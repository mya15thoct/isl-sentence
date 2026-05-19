"""
Sentence Classifier for ISL Recognition (Approach D — v2).

Improvements over v1:
  1. Multi-Head Self-Attention after BiLSTM — captures long-range dependencies
     in sentence videos (80-200 frames) that BiLSTM alone misses.
  2. Temporal Attention pooling (transferred from word model) — replaces masked
     mean pooling; learns which frames matter most instead of equal weighting.
  3. Dropout 0.4 → 0.6 on classifier head — reduces overfitting with small data.
  4. L2 regularization on Dense layers — further prevents overfitting.

Architecture:
  Input (B, T, 1662)
    → MLP branches (pose/face/hand)       [transfer from word model]
    → shared Dense layers                  [transfer from word model]
    → BiLSTM(64) → BiLSTM(32)             [transfer from word model]  → (B, T, 64)
    → MultiHeadAttention(4h, key=16)       [NEW — long sequence modeling]
    → LayerNorm + residual
    → Temporal Attention                   [transfer attn_td/temporal_attention]
    → context_vector                       (B, 64)
    → Dense(128, L2) + Dropout(0.6)
    → Dense(num_sentences, softmax)

Transfer layers (name-matched from word model):
  pose_features, face_features, hand_features,
  shared_td1, shared_td2, bilstm1, bilstm2,
  attn_td, temporal_attention, attn_apply, context_vector
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHECKPOINT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# MLP BRANCHES  (identical names to word model for weight transfer)
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
    Build sentence classifier with temporal attention and MHA.

    Args:
        num_sentences:   Number of sentence classes (e.g. 99)
        word_model_path: Path to best_model_combined (SavedModel)
        freeze_encoder:  If True, freeze MLP + BiLSTM layers (word model weights)
                         during phase 1, train only MHA + attention + head.

    Returns:
        Keras Model ready for compilation
    """
    reg = tf.keras.regularizers.l2(1e-4)

    # ── Encoder ──────────────────────────────────────────────────────────────
    inputs = layers.Input(shape=(None, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_kp = layers.Lambda(lambda t: t[:, :, :132],     name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],    name='hand_split')(x)

    pose_feat = layers.TimeDistributed(_pose_branch(132,  'pose'), name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(_face_branch(1404, 'face'), name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(_hand_branch(126,  'hand'), name='hand_features')(hand_kp)

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

    # ── Multi-Head Self-Attention (NEW) ───────────────────────────────────────
    # Captures long-range dependencies across frames — BiLSTM sees context
    # sequentially, MHA sees all frame pairs simultaneously.
    mha_out = layers.MultiHeadAttention(
        num_heads=4, key_dim=16, dropout=0.1, name='mha')(x, x)
    x = layers.Add(name='mha_residual')([x, mha_out])       # residual connection
    x = layers.LayerNormalization(name='mha_ln')(x)          # (B, T, 64)

    # ── Temporal Attention pooling ────────────────────────────────────────────
    # Learns which frames are most informative for sentence identity.
    # Layer names match word model exactly → weights are transferred.
    scores  = layers.TimeDistributed(
        layers.Dense(1, name='attn_dense'), name='attn_td')(x)       # (B, T, 1)
    weights = layers.Softmax(axis=1, name='temporal_attention')(scores)  # (B, T, 1)
    x       = layers.Multiply(name='attn_apply')([x, weights])           # (B, T, 64)
    x       = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1), name='context_vector')(x)    # (B, 64)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dense(
        128, activation='relu',
        kernel_regularizer=reg, name='cls_dense')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(
        num_sentences, activation='softmax', name='sentence_probs')(x)

    model = Model(inputs=inputs, outputs=outputs, name='SentenceClassifier_v2')

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

    # ── Freeze encoder (phase 1 only) ─────────────────────────────────────────
    # Freeze only MLP + BiLSTM (heavy word model weights).
    # MHA + temporal attention + head remain trainable so they can adapt
    # to sentence-level patterns from the start.
    if freeze_encoder:
        encoder_layers = {
            'pose_features', 'face_features', 'hand_features',
            'shared_td1', 'shared_td2', 'bilstm1', 'bilstm2',
        }
        for layer in model.layers:
            if layer.name in encoder_layers:
                layer.trainable = False
        print('[Classifier] MLP + BiLSTM frozen — MHA + attention + head will train')

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
    print(f'Output shape : {out.shape}')
    assert out.shape == (4, NUM_SENTENCES)
    print('[OK] Output shape correct')
