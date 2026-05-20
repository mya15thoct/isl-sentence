"""
Skeleton GCN for static sign classification.

Models the human skeleton as a graph:
  - Nodes: joints (pose=33, left_hand=21, right_hand=21)
  - Edges: anatomical connections (bones)

Input: single frame keypoint (1662,) → reshape to joints
Output: gloss class (softmax)

Why GCN > MLP for skeleton:
  MLP treats 1662 values as flat vector — ignores spatial structure.
  GCN propagates information along skeleton edges — thumb knows what
  index finger is doing, wrist knows what elbow is doing.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SKELETON GRAPH DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

# MediaPipe Pose connections (33 joints, 4 values each: x,y,z,vis)
POSE_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),
]

# MediaPipe Hand connections (21 joints each, 3 values: x,y,z)
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

NUM_POSE  = 33
NUM_HAND  = 21
NUM_NODES = NUM_POSE + NUM_HAND + NUM_HAND   # 75 joints total


def build_adjacency() -> np.ndarray:
    """Build normalized adjacency matrix for pose + left_hand + right_hand."""
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    # Pose edges
    for i, j in POSE_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    # Left hand edges (offset = 33)
    lh_off = NUM_POSE
    for i, j in HAND_EDGES:
        A[lh_off+i, lh_off+j] = 1.0
        A[lh_off+j, lh_off+i] = 1.0

    # Right hand edges (offset = 33 + 21 = 54)
    rh_off = NUM_POSE + NUM_HAND
    for i, j in HAND_EDGES:
        A[rh_off+i, rh_off+j] = 1.0
        A[rh_off+j, rh_off+i] = 1.0

    # Connect wrists (pose 15=left_wrist, 16=right_wrist) to hand roots
    A[15, lh_off] = 1.0; A[lh_off, 15] = 1.0
    A[16, rh_off] = 1.0; A[rh_off, 16] = 1.0

    # Self-loops
    np.fill_diagonal(A, 1.0)

    # Symmetric row normalization: D^{-1/2} A D^{-1/2}
    deg  = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    A = (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    return A


# ─────────────────────────────────────────────────────────────────────────────
# GCN LAYER
# ─────────────────────────────────────────────────────────────────────────────

class GCNLayer(layers.Layer):
    def __init__(self, units: int, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.activation = activation
        self.dense      = layers.Dense(units, use_bias=False)
        self.bn         = layers.BatchNormalization()

    def call(self, x, A, training=False):
        # x: (B, N, F), A: (N, N)
        out = tf.matmul(A, x)                      # (B, N, F) — aggregate neighbours
        out = self.dense(out)                       # (B, N, units)
        out = self.bn(out, training=training)
        if self.activation == 'relu':
            out = tf.nn.relu(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units, 'activation': self.activation})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_skeleton_gcn(num_classes: int, dropout: float = 0.5) -> Model:
    """
    Input: (B, 1662) flat keypoint vector
    Reshape to (B, 75, F) node features, apply GCN, classify.

    Node features:
      pose joints  (0-32) : x, y, z, visibility  → 4 features
      left hand    (33-53): x, y, z               → 3 features
      right hand   (54-74): x, y, z               → 3 features
    Padded to uniform 4 features per node.
    """
    A_np = build_adjacency()
    A    = tf.constant(A_np, dtype=tf.float32)   # (75, 75)

    inputs = layers.Input(shape=(1662,), name='frame_input')

    # Split into joints
    pose_flat = layers.Lambda(lambda t: t[:, :132])(inputs)         # (B, 132) = 33*4
    lh_flat   = layers.Lambda(lambda t: t[:, 1536:1599])(inputs)    # (B, 63)  = 21*3
    rh_flat   = layers.Lambda(lambda t: t[:, 1599:])(inputs)        # (B, 63)  = 21*3

    # Reshape to (B, joints, features)
    pose_nodes = layers.Reshape((33, 4))(pose_flat)                  # (B, 33, 4)

    # Pad hand joints to 4 features (append zero for visibility)
    lh_nodes   = layers.Reshape((21, 3))(lh_flat)
    rh_nodes   = layers.Reshape((21, 3))(rh_flat)
    lh_nodes   = layers.ZeroPadding2D(padding=((0,0),(0,1)))(
                     layers.Reshape((21, 3, 1))(lh_flat))            # hack via reshape
    # Simpler: use Dense to project 3→4
    lh_nodes   = layers.Reshape((21, 3))(lh_flat)
    lh_nodes   = layers.TimeDistributed(layers.Dense(4, use_bias=False))(lh_nodes)
    rh_nodes   = layers.Reshape((21, 3))(rh_flat)
    rh_nodes   = layers.TimeDistributed(layers.Dense(4, use_bias=False))(rh_nodes)

    # Concatenate all nodes: (B, 75, 4)
    node_feat = layers.Concatenate(axis=1)([pose_nodes, lh_nodes, rh_nodes])

    # GCN layers
    gcn1 = GCNLayer(64,  name='gcn1')
    gcn2 = GCNLayer(128, name='gcn2')
    gcn3 = GCNLayer(256, name='gcn3')

    x = gcn1(node_feat, A)   # (B, 75, 64)
    x = gcn2(x, A)            # (B, 75, 128)
    x = gcn3(x, A)            # (B, 75, 256)

    # Global mean pooling over nodes
    x = layers.GlobalAveragePooling1D()(x)   # (B, 256)

    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout * 0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='gloss_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='SkeletonGCN')
