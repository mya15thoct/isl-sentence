"""
Keypoint preprocessing utilities for ISL gloss pipeline.

Shoulder normalization is already applied during .npy extraction.
This module adds wrist-relative normalization for hand joints.
"""

import numpy as np


# MediaPipe hand layout: joint 0=wrist, joint 12=middle fingertip
_LH_START  = 1536
_RH_START  = 1599
_HAND_DIMS = 63   # 21 joints * 3


def normalize_hands_wrist(kp: np.ndarray) -> np.ndarray:
    """
    Normalize hand joints relative to wrist position and hand size.

    For each hand:
      1. Subtract wrist (joint 0) → hand-centered coordinates
      2. Divide by wrist-to-middle-fingertip distance → scale-invariant

    Hands with all-zero keypoints (not detected) are left untouched.
    """
    kp = kp.copy()

    for start in (_LH_START, _RH_START):
        hand = kp[start:start + _HAND_DIMS].reshape(21, 3)

        # Skip if hand not detected
        if np.all(hand == 0):
            continue

        wrist = hand[0].copy()          # joint 0 = wrist
        hand  = hand - wrist            # center at wrist

        # Scale by wrist-to-middle-fingertip distance (joint 12)
        scale = np.linalg.norm(hand[12])
        if scale > 1e-6:
            hand = hand / scale

        kp[start:start + _HAND_DIMS] = hand.flatten()

    return kp


def apply_all(kp: np.ndarray) -> np.ndarray:
    """Apply all preprocessing steps after shoulder normalization."""
    return normalize_hands_wrist(kp)
