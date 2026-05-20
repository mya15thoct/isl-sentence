"""
Joint Transformer for static ISL sign classification.

Treats each skeleton joint as a token → Multi-Head Self-Attention
captures global relationships between ALL joint pairs simultaneously.

Joints (75 total):
  0-32  : pose joints    (x, y, z, visibility) → project to d_model
  33-53 : left hand      (x, y, z)             → project to d_model
  54-74 : right hand     (x, y, z)             → project to d_model

Why better than MLP:
  MLP: flat 1662-dim vector, ignores structure
  GCN: only local neighbourhood (fixed graph)
  Transformer: all-to-all attention — thumb knows what opposite wrist is doing
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


NUM_POSE  = 33
NUM_HAND  = 21
NUM_NODES = NUM_POSE + NUM_HAND + NUM_HAND   # 75


class JointEmbedding(layers.Layer):
    """Project each joint's coordinates to d_model + add learnable positional encoding."""
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model    = d_model
        self.pose_proj  = layers.Dense(d_model)
        self.hand_proj  = layers.Dense(d_model)
        self.pos_embed  = self.add_weight(
            shape=(1, NUM_NODES, d_model),
            initializer='glorot_uniform',
            trainable=True,
            name='pos_embed',
        )

    def call(self, inputs):
        # inputs: (B, 1662)
        pose_flat = inputs[:, :132]                # (B, 132) = 33*4
        lh_flat   = inputs[:, 1536:1599]           # (B, 63)  = 21*3
        rh_flat   = inputs[:, 1599:]               # (B, 63)  = 21*3

        pose = tf.reshape(pose_flat, (-1, NUM_POSE, 4))    # (B, 33, 4)
        lh   = tf.reshape(lh_flat,   (-1, NUM_HAND, 3))    # (B, 21, 3)
        rh   = tf.reshape(rh_flat,   (-1, NUM_HAND, 3))    # (B, 21, 3)

        pose_emb = self.pose_proj(pose)   # (B, 33, d_model)
        lh_emb   = self.hand_proj(lh)     # (B, 21, d_model)
        rh_emb   = self.hand_proj(rh)     # (B, 21, d_model) — shared projection

        x = tf.concat([pose_emb, lh_emb, rh_emb], axis=1)  # (B, 75, d_model)
        return x + self.pos_embed                            # + positional encoding


class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ff1  = layers.Dense(ff_dim, activation='gelu')
        self.ff2  = layers.Dense(d_model)
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Multi-head self-attention + residual
        attn = self.mha(x, x, training=training)
        x    = self.ln1(x + self.drop(attn, training=training))
        # Feed-forward + residual
        ff   = self.ff2(self.ff1(x))
        x    = self.ln2(x + self.drop(ff, training=training))
        return x


def build_joint_transformer(
    num_classes: int,
    d_model:     int   = 128,
    num_heads:   int   = 4,
    ff_dim:      int   = 256,
    num_layers:  int   = 4,
    dropout:     float = 0.3,
) -> Model:
    inputs = layers.Input(shape=(1662,), name='frame_input')

    # Joint embedding
    x = JointEmbedding(d_model, name='joint_embed')(inputs)   # (B, 75, d_model)
    x = layers.Dropout(dropout)(x)

    # Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(
            d_model, num_heads, ff_dim, dropout,
            name=f'transformer_{i}')(x)

    # Pool over joints — use both mean and max
    mean_pool = layers.GlobalAveragePooling1D()(x)   # (B, d_model)
    max_pool  = layers.GlobalMaxPooling1D()(x)        # (B, d_model)
    x         = layers.Concatenate()([mean_pool, max_pool])  # (B, 2*d_model)

    # Classification head
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dropout(dropout * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='gloss_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='JointTransformer')
