"""
Cross-Branch Attention + ArcFace loss for ISL sign classification.

ArcFace replaces softmax with angular margin loss:
  - L2-normalize embeddings and weight matrix
  - Add angular margin m to correct class angle
  - Forces same-class embeddings to cluster tightly
  - Proven for few-shot recognition (face recognition, small datasets)

Inference interface identical to standard model:
  model(x) → softmax probabilities (B, C)  ← MLPDecoder compatible
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class ArcFaceLayer(layers.Layer):
    """L2-normalized weight matrix for cosine similarity classification."""
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer='glorot_uniform',
            trainable=True,
            name='arcface_W',
        )

    def call(self, x):
        x_norm = tf.nn.l2_normalize(x, axis=1)          # (B, emb_dim)
        W_norm = tf.nn.l2_normalize(self.W, axis=0)     # (emb_dim, C)
        return tf.matmul(x_norm, W_norm)                 # (B, C) cosine similarities


class CrossBranchArcFace(tf.keras.Model):
    """
    Cross-Branch backbone + ArcFace loss.

    train_step: applies angular margin → better embedding separation
    call / test_step: softmax(s * cosine) → probabilities (MLPDecoder compatible)
    """
    def __init__(self, backbone: Model, num_classes: int,
                 s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.backbone    = backbone
        self.arcface     = ArcFaceLayer(num_classes, name='arcface')
        self.s           = s
        self.m           = m
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_tracker  = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    def call(self, x, training=False):
        emb    = self.backbone(x, training=training)
        cos    = self.arcface(emb)                       # (B, C) cosine similarities
        return tf.nn.softmax(self.s * cos)               # probabilities for inference

    def train_step(self, data):
        X, y = data
        y_int = tf.cast(tf.reshape(y, [-1]), tf.int32)

        with tf.GradientTape() as tape:
            emb = self.backbone(X, training=True)
            cos_theta = self.arcface(emb)                # (B, C)

            # Angular margin: add m to correct class angle
            one_hot    = tf.one_hot(y_int, tf.shape(cos_theta)[1])
            theta      = tf.acos(tf.clip_by_value(cos_theta, -1 + 1e-7, 1 - 1e-7))
            cos_theta_m = tf.cos(theta + self.m)

            logits = self.s * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)
            loss   = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_int, logits, from_logits=True))

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y_int, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, y = data
        y_int = tf.cast(tf.reshape(y, [-1]), tf.int32)
        probs = self(X, training=False)
        loss  = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_int, probs))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y_int, probs)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]


def build_arcface_model(
    num_classes: int,
    d_model:     int   = 128,
    num_heads:   int   = 4,
    ff_dim:      int   = 256,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    s:           float = 30.0,
    m:           float = 0.5,
) -> CrossBranchArcFace:
    from gloss_pipeline.models.cross_branch_attn import (
        BranchEncoder, CrossBranchBlock)

    inputs = layers.Input(shape=(1662,), name='frame_input')

    pose = inputs[:, :132]
    face = inputs[:, 132:1536]
    lh   = inputs[:, 1536:1599]
    rh   = inputs[:, 1599:]

    pose_emb = BranchEncoder(d_model, dropout, name='pose_enc')(pose)
    face_emb = BranchEncoder(d_model, dropout, name='face_enc')(face)
    lh_emb   = BranchEncoder(d_model, dropout, name='lh_enc')(lh)
    rh_emb   = BranchEncoder(d_model, dropout, name='rh_enc')(rh)

    x = layers.Lambda(lambda t: tf.stack(t, axis=1),
                      name='stack_branches')([pose_emb, face_emb, lh_emb, rh_emb])
    x = layers.Dropout(dropout, name='branch_dropout')(x)

    for i in range(num_layers):
        x = CrossBranchBlock(d_model, num_heads, ff_dim, dropout,
                             name=f'cross_attn_{i}')(x)

    mean_pool = layers.GlobalAveragePooling1D(name='mean_pool')(x)
    max_pool  = layers.GlobalMaxPooling1D(name='max_pool')(x)
    x         = layers.Concatenate(name='pool_concat')([mean_pool, max_pool])

    # Embedding head (no softmax — ArcFace handles classification)
    x = layers.Dense(256, activation='gelu', name='head_fc1')(x)
    x = layers.Dropout(dropout, name='head_drop1')(x)
    x = layers.Dense(128, name='embedding')(x)   # L2-normalized by ArcFaceLayer

    backbone = Model(inputs=inputs, outputs=x, name='CrossBranchBackbone')
    return CrossBranchArcFace(backbone, num_classes, s=s, m=m)
