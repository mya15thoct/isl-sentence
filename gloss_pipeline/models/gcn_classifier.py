"""
GCN classifier on top of word model BiLSTM features.

Architecture:
  Input (window, 1662)
    → word model feature extractor (frozen bilstm2) → (T, 64)  [motion dynamics]
    → Graph Convolution over temporal nodes                      [spatial relationships]
    → Temporal Attention pooling → (64,)
    → Dense → num_classes

The word model (video-only, 97%) is frozen — used only to extract motion dynamics.
GCN + head are trained on ISL-Frames-Data pseudo-sequences.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import VIDEO_WORD_MODEL_PATH


class GraphConvolution(layers.Layer):
    """
    Simple graph convolution: H' = A H W
    A: adjacency matrix (num_nodes, num_nodes)
    H: node features   (B, num_nodes, features)
    W: learnable weight
    """
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units, use_bias=True)

    def call(self, inputs, adjacency):
        # inputs: (B, N, F), adjacency: (N, N)
        agg = tf.matmul(adjacency, inputs)   # (B, N, F)
        return tf.nn.relu(self.dense(agg))   # (B, N, units)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg


def _build_temporal_adjacency(seq_len: int) -> tf.Tensor:
    """
    Temporal graph: each time step connected to itself and neighbours.
    A[i,j] = 1 if |i-j| <= 1, normalized by degree.
    """
    A = tf.eye(seq_len)
    if seq_len > 1:
        off = tf.linalg.diag(tf.ones(seq_len - 1), k=1) + \
              tf.linalg.diag(tf.ones(seq_len - 1), k=-1)
        A = A + off
    # Row-normalize
    deg = tf.reduce_sum(A, axis=1, keepdims=True)
    return A / tf.maximum(deg, 1.0)


def build_gcn_classifier(
    num_classes:      int,
    word_model_path:  str  = None,
    seq_len:          int  = 30,
    freeze_extractor: bool = True,
) -> Model:
    word_model_path = word_model_path or str(VIDEO_WORD_MODEL_PATH)

    # ── Load word model and build feature extractor ───────────────────────────
    print(f'[GCN] Loading word model from: {word_model_path}')
    word_model = tf.keras.models.load_model(word_model_path)

    feature_extractor = Model(
        inputs  = word_model.input,
        outputs = word_model.get_layer('bilstm2').output,   # (B, T, 64)
        name    = 'motion_feature_extractor',
    )
    if freeze_extractor:
        feature_extractor.trainable = False
        print('[GCN] Feature extractor frozen')

    # ── Build GCN on top ──────────────────────────────────────────────────────
    # Fixed-length input (seq_len, 1662) for training on pseudo-sequences
    inputs = layers.Input(shape=(seq_len, 1662), name='sequence_input')

    # Extract motion dynamic features from word model
    x = feature_extractor(inputs)    # (B, seq_len, 64)

    # Temporal adjacency matrix
    A = _build_temporal_adjacency(seq_len)   # (seq_len, seq_len)

    # Graph convolution layers
    gc1 = GraphConvolution(64,  name='gc1')
    gc2 = GraphConvolution(128, name='gc2')

    x = gc1(x, A)    # (B, seq_len, 64)
    x = gc2(x, A)    # (B, seq_len, 128)

    # Temporal attention pooling
    scores  = layers.TimeDistributed(layers.Dense(1), name='attn_td')(x)
    weights = layers.Softmax(axis=1, name='attn_weights')(scores)
    x       = layers.Multiply()([x, weights])
    x       = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)  # (B, 128)

    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='gloss_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='GCN_ISL')
