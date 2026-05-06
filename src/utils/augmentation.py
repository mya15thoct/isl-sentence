"""
Data augmentation utilities for sign language sequences
"""
import numpy as np
from typing import Tuple


def temporal_reverse(sequence: np.ndarray) -> np.ndarray:
    return np.flip(sequence, axis=0).copy()


def temporal_subsample(sequence: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    from scipy.interpolate import interp1d
    num_frames = len(sequence)
    num_keep = max(int(num_frames * ratio), 2)
    indices = sorted(np.random.choice(num_frames, num_keep, replace=False))
    interpolator = interp1d(indices, sequence[indices], axis=0,
                            kind='linear', fill_value='extrapolate')
    return interpolator(np.arange(num_frames)).astype(np.float32)


def add_noise(sequence: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, noise_std, sequence.shape)
    return (sequence + noise).astype(np.float32)


def spatial_jitter(sequence: np.ndarray, jitter_std: float = 0.02) -> np.ndarray:
    """
    Add small random spatial offset to all x, y coordinates each frame.
    Pose: stride 4 (x,y,z,vis) | Face/Hand: stride 3 (x,y,z)
    """
    seq = sequence.copy()
    for frame_idx in range(len(seq)):
        if np.all(seq[frame_idx] == 0):
            continue
        dx = np.random.normal(0, jitter_std)
        dy = np.random.normal(0, jitter_std)
        seq[frame_idx, 0:132:4]   += dx
        seq[frame_idx, 1:132:4]   += dy
        seq[frame_idx, 132:1536:3] += dx
        seq[frame_idx, 133:1536:3] += dy
        seq[frame_idx, 1536::3]   += dx
        seq[frame_idx, 1537::3]   += dy
    return seq.astype(np.float32)
