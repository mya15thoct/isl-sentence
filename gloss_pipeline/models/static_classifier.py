"""
MLP static classifier for ISL-Frames-Data words.
Input: single frame keypoint (1662-dim)
Output: gloss class (softmax)

No BiLSTM — static signs are identified by hand shape + position alone.
MLP branch names match word model for optional weight transfer.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_static_classifier(num_classes: int) -> Model:
    inputs = layers.Input(shape=(1662,), name='frame_input')

    pose_kp = layers.Lambda(lambda t: t[:, :132],    name='pose_split')(inputs)
    face_kp = layers.Lambda(lambda t: t[:, 132:1536], name='face_split')(inputs)
    hand_kp = layers.Lambda(lambda t: t[:, 1536:],   name='hand_split')(inputs)

    # Pose branch
    p = layers.Dense(128, activation='relu')(pose_kp)
    p = layers.BatchNormalization()(p)
    p = layers.Dense(64,  activation='relu')(p)
    p = layers.BatchNormalization()(p)

    # Face branch
    f = layers.Dense(256, activation='relu')(face_kp)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.3)(f)
    f = layers.Dense(128, activation='relu')(f)
    f = layers.BatchNormalization()(f)

    # Hand branch
    h = layers.Dense(128, activation='relu')(hand_kp)
    h = layers.BatchNormalization()(h)
    h = layers.Dense(64,  activation='relu')(h)
    h = layers.BatchNormalization()(h)

    x = layers.Concatenate()([p, f, h])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='gloss_probs')(x)

    return Model(inputs=inputs, outputs=outputs, name='StaticClassifier')
