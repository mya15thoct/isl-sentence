"""
Sentence Model for Continuous ISL Recognition (CTC-based).

Architecture:
  - Reuses encoder weights from best_model_combined (transfer learning)
  - Same MLP branches + BiLSTM as hybrid.py
  - Removes temporal attention (which collapses T→1)
  - Adds Dense(num_glosses+1) CTC output head per frame

Input:  (batch, T, 1662)   ← variable T per sentence video
Output: (batch, T, num_glosses+1)  ← CTC blank token at index 0

Usage:
  model = build_sentence_model(
      num_glosses=409,
      word_model_path='/mnt/ngan/ISL-Sequences/checkpoints/best_model_combined'
  )
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHECKPOINT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# MLP BRANCHES  (same as vsl-recognition/src/models/components.py)
# Inlined here to avoid cross-project dependency
# ─────────────────────────────────────────────────────────────────────────────

def _pose_branch(input_dim: int, name: str):
    """Shallow 2-layer MLP for pose keypoints → 64 dims."""
    return tf.keras.Sequential([
        layers.Dense(128, activation='relu', name=f'{name}_d1'),
        layers.BatchNormalization(name=f'{name}_bn1'),
        layers.Dense(64,  activation='relu', name=f'{name}_d2'),
        layers.BatchNormalization(name=f'{name}_bn2'),
    ], name=name)


def _face_branch(input_dim: int, name: str):
    """Deep 4-layer MLP for face keypoints → 128 dims."""
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
    """Deep 3-layer MLP for hand keypoints → 64 dims."""
    return tf.keras.Sequential([
        layers.Dense(256, activation='relu', name=f'{name}_d1'),
        layers.BatchNormalization(name=f'{name}_bn1'),
        layers.Dense(128, activation='relu', name=f'{name}_d2'),
        layers.BatchNormalization(name=f'{name}_bn2'),
        layers.Dense(64,  activation='relu', name=f'{name}_d3'),
        layers.BatchNormalization(name=f'{name}_bn3'),
    ], name=name)


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def _build_encoder(num_glosses: int) -> Model:
    """
    Build the sentence encoder:
      MLP branches → concat → shared layers → BiLSTM × 2 → Dense(CTC head)

    Key differences from hybrid.py:
      - Input sequence_length = None  (variable T for sentence videos)
      - Temporal Attention REMOVED  (it collapses T→1, incompatible with CTC)
      - Output: (B, T, num_glosses+1)  instead of (B, num_classes)
    """
    # === INPUT ===
    inputs = layers.Input(shape=(None, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    # === SPLIT KEYPOINTS ===
    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    # === MLP BRANCHES ===
    pose_branch = _pose_branch(132,  'pose')
    face_branch = _face_branch(1404, 'face')
    hand_branch = _hand_branch(126,  'hand')

    pose_features = layers.TimeDistributed(pose_branch, name='pose_features')(pose_kp)   # (B,T,64)
    face_features = layers.TimeDistributed(face_branch, name='face_features')(face_kp)   # (B,T,128)
    hand_features = layers.TimeDistributed(hand_branch, name='hand_features')(hand_kp)   # (B,T,64)

    # === FEATURE FUSION ===
    merged = layers.Concatenate(name='feature_fusion')([pose_features, face_features, hand_features])  # (B,T,256)

    # === SHARED LAYERS ===
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(0.3)(x)

    # === BIDIRECTIONAL LSTM ===
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bilstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name='bilstm2')(x)  # (B,T,64)
    x = layers.Dropout(0.3)(x)

    # === CTC OUTPUT HEAD ===
    # num_glosses + 1: index 0 reserved for CTC blank token
    ctc_logits = layers.Dense(num_glosses + 1, activation='linear', name='ctc_logits')(x)
    outputs    = layers.Softmax(name='ctc_out')(ctc_logits)  # (B, T, num_glosses+1)

    return Model(inputs=inputs, outputs=outputs, name='SentenceModel_CTC')


def build_sentence_model(
    num_glosses:    int,
    word_model_path: str,
    freeze_encoder: bool = False,
) -> Model:
    """
    Build and initialise the Sentence Model.

    Args:
        num_glosses:     Number of ISL gloss classes (from action_mapping_combined)
        word_model_path: Path to best_model_combined (SavedModel or H5)
        freeze_encoder:  If True, freeze all layers except CTC head

    Returns:
        Keras Model ready for CTC training
    """
    print('[SentenceModel] Building architecture...')
    model = _build_encoder(num_glosses)

    print(f'[SentenceModel] Loading word model from: {word_model_path}')
    try:
        word_model = tf.keras.models.load_model(str(word_model_path))
    except Exception as e:
        print(f'[WARNING] Could not load word model: {e}')
        print('          Proceeding with random initialisation.')
        return model

    # Transfer weights layer by layer (name-based matching)
    transferred, skipped = 0, 0
    for layer in model.layers:
        try:
            word_layer = word_model.get_layer(layer.name)
            weights    = word_layer.get_weights()
            if weights:
                layer.set_weights(weights)
                transferred += 1
        except Exception:
            skipped += 1

    print(f'[SentenceModel] Weights transferred: {transferred}  |  Skipped (new): {skipped}')

    if freeze_encoder:
        for layer in model.layers:
            if layer.name != 'ctc_logits' and layer.name != 'ctc_out':
                layer.trainable = False
        print('[SentenceModel] Encoder frozen — only CTC head will be trained')

    return model


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np

    NUM_GLOSSES = 409
    model = _build_encoder(NUM_GLOSSES)
    model.summary()

    # Simulate a batch of 2 sentence videos with different lengths (padded)
    T = 200
    x = np.random.rand(2, T, 1662).astype(np.float32)
    out = model(x, training=False)

    print(f'\nInput  shape: {x.shape}')
    print(f'Output shape: {out.shape}')   # (2, T, 410)
    assert out.shape == (2, T, NUM_GLOSSES + 1), 'Shape mismatch!'
    print('[OK] Output shape correct')
