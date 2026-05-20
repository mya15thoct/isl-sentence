"""
Cross-Branch Attention classifier for static ISL sign recognition.

Architecture:
  Input (1662,)
    → split into 4 branches: pose (132), face (1404), left hand (63), right hand (63)
    → each branch → BranchEncoder (2-layer MLP + LayerNorm) → (d_model,)
    → stack branches → (4, d_model)
    → Multi-Head Self-Attention across branches (captures inter-branch relationships)
    → mean+max pool → (2*d_model,)
    → classification head → num_classes
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


class BranchEncoder(layers.Layer):
    """Two-layer MLP + LayerNorm to encode a branch's raw keypoints."""
    def __init__(self, d_model: int, dropout: float = 0.1,
                 l2_reg: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        reg = l2(l2_reg)
        self.fc1  = layers.Dense(d_model * 2, activation='gelu',
                                 kernel_regularizer=reg)
        self.fc2  = layers.Dense(d_model, kernel_regularizer=reg)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        return self.norm(x)


class CrossBranchBlock(layers.Layer):
    """One MHA block with residual + feedforward across branch tokens."""
    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, l2_reg: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        reg = l2(l2_reg)
        self.mha  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ff1  = layers.Dense(ff_dim, activation='gelu',
                                 kernel_regularizer=reg)
        self.ff2  = layers.Dense(d_model, kernel_regularizer=reg)
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x    = self.ln1(x + self.drop(attn, training=training))
        ff   = self.ff2(self.ff1(x))
        x    = self.ln2(x + self.drop(ff, training=training))
        return x


def build_cross_branch_attn(
    num_classes: int,
    d_model:     int   = 64,
    num_heads:   int   = 4,
    ff_dim:      int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.5,
    l2_reg:      float = 1e-4,
) -> Model:
    inputs = layers.Input(shape=(1662,), name='frame_input')

    # ── Split into branches ───────────────────────────────────────────────────
    pose = inputs[:, :132]          # (B, 132)
    face = inputs[:, 132:1536]      # (B, 1404)
    lh   = inputs[:, 1536:1599]     # (B, 63)
    rh   = inputs[:, 1599:]         # (B, 63)

    # ── Branch encoders ───────────────────────────────────────────────────────
    pose_emb = BranchEncoder(d_model, dropout, l2_reg, name='pose_enc')(pose)
    face_emb = BranchEncoder(d_model, dropout, l2_reg, name='face_enc')(face)
    lh_emb   = BranchEncoder(d_model, dropout, l2_reg, name='lh_enc')(lh)
    rh_emb   = BranchEncoder(d_model, dropout, l2_reg, name='rh_enc')(rh)

    # Stack → (B, 4, d_model)
    x = layers.Lambda(
        lambda t: tf.stack(t, axis=1),
        name='stack_branches',
    )([pose_emb, face_emb, lh_emb, rh_emb])

    x = layers.Dropout(dropout, name='branch_dropout')(x)

    # ── Cross-branch attention blocks ─────────────────────────────────────────
    for i in range(num_layers):
        x = CrossBranchBlock(d_model, num_heads, ff_dim, dropout, l2_reg,
                             name=f'cross_attn_{i}')(x)

    # ── Pool ──────────────────────────────────────────────────────────────────
    mean_pool = layers.GlobalAveragePooling1D(name='mean_pool')(x)
    max_pool  = layers.GlobalMaxPooling1D(name='max_pool')(x)
    x         = layers.Concatenate(name='pool_concat')([mean_pool, max_pool])

    # ── Classification head ───────────────────────────────────────────────────
    reg = l2(l2_reg)
    x = layers.Dense(128, activation='gelu', kernel_regularizer=reg,
                     name='head_fc1')(x)
    x = layers.Dropout(dropout, name='head_drop1')(x)
    x = layers.Dense(64, activation='gelu', kernel_regularizer=reg,
                     name='head_fc2')(x)
    x = layers.Dropout(dropout * 0.5, name='head_drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='gloss_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='CrossBranchAttn')
