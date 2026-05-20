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

Advantages over Joint Transformer:
  - Only 4 tokens (branches) vs 75 tokens (joints) → much less parameters
  - Cross-branch attention captures: how hands relate to pose, LH vs RH coordination
  - Same (B, 1662) input interface → drop-in for MLPDecoder
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class BranchEncoder(layers.Layer):
    """Two-layer MLP + LayerNorm to encode a branch's raw keypoints."""
    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.fc1  = layers.Dense(d_model * 2, activation='gelu')
        self.fc2  = layers.Dense(d_model)
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
        attn = self.mha(x, x, training=training)
        x    = self.ln1(x + self.drop(attn, training=training))
        ff   = self.ff2(self.ff1(x))
        x    = self.ln2(x + self.drop(ff, training=training))
        return x


def build_cross_branch_attn(
    num_classes: int,
    d_model:     int   = 128,
    num_heads:   int   = 4,
    ff_dim:      int   = 256,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
) -> Model:
    """
    Build Cross-Branch Attention model.

    Input: (B, 1662) — shoulder-normalized keypoints
    Output: (B, num_classes) softmax probabilities
    """
    inputs = layers.Input(shape=(1662,), name='frame_input')

    # ── Split into branches ───────────────────────────────────────────────────
    pose = inputs[:, :132]          # (B, 132) = 33 joints * 4
    face = inputs[:, 132:1536]      # (B, 1404) = 468 face landmarks * 3
    lh   = inputs[:, 1536:1599]     # (B, 63)  = 21 left hand joints * 3
    rh   = inputs[:, 1599:]         # (B, 63)  = 21 right hand joints * 3

    # ── Branch encoders → (B, d_model) each ──────────────────────────────────
    pose_emb = BranchEncoder(d_model, dropout, name='pose_enc')(pose)
    face_emb = BranchEncoder(d_model, dropout, name='face_enc')(face)
    lh_emb   = BranchEncoder(d_model, dropout, name='lh_enc')(lh)
    rh_emb   = BranchEncoder(d_model, dropout, name='rh_enc')(rh)

    # Stack → (B, 4, d_model)  [4 branch tokens]
    x = layers.Lambda(
        lambda t: tf.stack(t, axis=1),
        name='stack_branches',
    )([pose_emb, face_emb, lh_emb, rh_emb])

    x = layers.Dropout(dropout, name='branch_dropout')(x)

    # ── Cross-branch attention blocks ─────────────────────────────────────────
    for i in range(num_layers):
        x = CrossBranchBlock(d_model, num_heads, ff_dim, dropout,
                             name=f'cross_attn_{i}')(x)

    # ── Pool over branch tokens ───────────────────────────────────────────────
    mean_pool = layers.GlobalAveragePooling1D(name='mean_pool')(x)   # (B, d_model)
    max_pool  = layers.GlobalMaxPooling1D(name='max_pool')(x)         # (B, d_model)
    x         = layers.Concatenate(name='pool_concat')([mean_pool, max_pool])

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.Dense(256, activation='gelu', name='head_fc1')(x)
    x = layers.Dropout(dropout, name='head_drop1')(x)
    x = layers.Dense(128, activation='gelu', name='head_fc2')(x)
    x = layers.Dropout(dropout * 0.5, name='head_drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='gloss_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='CrossBranchAttn')
