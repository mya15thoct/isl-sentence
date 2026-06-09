from __future__ import annotations

import random

import numpy as np


POSE_DIM = 132
FACE_DIM = 1404
HAND_DIM = 63
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM + HAND_DIM


def _indices_for_offsets(start: int, stop: int, stride: int, offsets: tuple[int, ...]) -> np.ndarray:
    chunks = [np.arange(start + offset, stop, stride, dtype=np.int64) for offset in offsets]
    return np.concatenate(chunks)


POSE_X = _indices_for_offsets(0, POSE_DIM, 4, (0,))
POSE_Y = _indices_for_offsets(0, POSE_DIM, 4, (1,))
FACE_X = _indices_for_offsets(POSE_DIM, POSE_DIM + FACE_DIM, 3, (0,))
FACE_Y = _indices_for_offsets(POSE_DIM, POSE_DIM + FACE_DIM, 3, (1,))
HAND_X = _indices_for_offsets(POSE_DIM + FACE_DIM, KEYPOINT_DIM, 3, (0,))
HAND_Y = _indices_for_offsets(POSE_DIM + FACE_DIM, KEYPOINT_DIM, 3, (1,))
X_INDICES = np.concatenate([POSE_X, FACE_X, HAND_X])
Y_INDICES = np.concatenate([POSE_Y, FACE_Y, HAND_Y])

POSE_COORDS = _indices_for_offsets(0, POSE_DIM, 4, (0, 1, 2))
FACE_COORDS = _indices_for_offsets(POSE_DIM, POSE_DIM + FACE_DIM, 3, (0, 1, 2))
HAND_COORDS = _indices_for_offsets(POSE_DIM + FACE_DIM, KEYPOINT_DIM, 3, (0, 1, 2))
COORD_INDICES = np.concatenate([POSE_COORDS, FACE_COORDS, HAND_COORDS])


def real_frame_mask(sequence: np.ndarray) -> np.ndarray:
    return np.any(np.abs(sequence) > 1e-8, axis=1)


def interpolate_sequence(sequence: np.ndarray, new_length: int) -> np.ndarray:
    old_length = len(sequence)
    if old_length == new_length:
        return sequence.astype(np.float32, copy=True)
    if old_length <= 1:
        return np.repeat(sequence, max(new_length, 1), axis=0).astype(np.float32)

    old_x = np.arange(old_length, dtype=np.float32)
    new_x = np.linspace(0, old_length - 1, max(new_length, 2), dtype=np.float32)
    out = np.empty((len(new_x), sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        out[:, dim] = np.interp(new_x, old_x, sequence[:, dim])
    return out


def interpolate_from_kept_frames(sequence: np.ndarray, kept_indices: np.ndarray) -> np.ndarray:
    if len(kept_indices) >= len(sequence):
        return sequence.astype(np.float32, copy=True)
    target_x = np.arange(len(sequence), dtype=np.float32)
    source_x = kept_indices.astype(np.float32)
    source = sequence[kept_indices]
    out = np.empty_like(sequence, dtype=np.float32)
    for dim in range(sequence.shape[1]):
        out[:, dim] = np.interp(target_x, source_x, source[:, dim])
    return out


def temporal_subsample(sequence: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    length = len(sequence)
    if length <= 2:
        return sequence.astype(np.float32, copy=True)
    keep = max(2, min(length, int(round(length * ratio))))
    indices = np.sort(np.random.choice(length, keep, replace=False))
    return interpolate_from_kept_frames(sequence, indices)


def temporal_scale(sequence: np.ndarray, scale: float = 0.9) -> np.ndarray:
    length = len(sequence)
    if length <= 2:
        return sequence.astype(np.float32, copy=True)
    new_length = max(2, int(round(length / scale)))
    scaled = interpolate_sequence(sequence, new_length)
    if len(scaled) == length:
        return scaled
    if len(scaled) > length:
        start = random.randint(0, len(scaled) - length)
        return scaled[start : start + length].astype(np.float32)
    pad = np.repeat(scaled[-1:], length - len(scaled), axis=0)
    return np.vstack([scaled, pad]).astype(np.float32)


def temporal_crop(sequence: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    length = len(sequence)
    if length <= 2:
        return sequence.astype(np.float32, copy=True)
    crop_length = max(2, min(length, int(round(length * crop_ratio))))
    start = random.randint(0, length - crop_length)
    return sequence[start : start + crop_length].astype(np.float32)


def add_coordinate_noise(sequence: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    seq = sequence.astype(np.float32, copy=True)
    mask = real_frame_mask(seq)
    if not np.any(mask):
        return seq
    noise = np.random.normal(0.0, noise_std, size=(int(mask.sum()), len(COORD_INDICES))).astype(
        np.float32
    )
    seq[np.ix_(mask, COORD_INDICES)] += noise
    return seq


def spatial_jitter(sequence: np.ndarray, jitter_std: float = 0.02) -> np.ndarray:
    seq = sequence.astype(np.float32, copy=True)
    mask = real_frame_mask(seq)
    frame_indices = np.where(mask)[0]
    for frame_idx in frame_indices:
        dx = np.float32(np.random.normal(0.0, jitter_std))
        dy = np.float32(np.random.normal(0.0, jitter_std))
        seq[frame_idx, X_INDICES] += dx
        seq[frame_idx, Y_INDICES] += dy
    return seq


def augment_sequence(
    sequence: np.ndarray,
    methods: list[str],
    probability: float = 0.75,
    noise_std: float = 0.01,
    jitter_std: float = 0.02,
    subsample_min: float = 0.75,
    subsample_max: float = 0.95,
    scale_min: float = 0.85,
    scale_max: float = 1.20,
    crop_min: float = 0.85,
    crop_max: float = 0.98,
) -> np.ndarray:
    if not methods or random.random() > probability:
        return sequence.astype(np.float32, copy=True)

    method = random.choice(methods)
    if method == "noise":
        out = add_coordinate_noise(sequence, noise_std=noise_std)
    elif method == "jitter":
        out = spatial_jitter(sequence, jitter_std=jitter_std)
    elif method == "subsample":
        out = temporal_subsample(sequence, ratio=random.uniform(subsample_min, subsample_max))
    elif method == "scale":
        out = temporal_scale(sequence, scale=random.uniform(scale_min, scale_max))
    elif method == "crop":
        out = temporal_crop(sequence, crop_ratio=random.uniform(crop_min, crop_max))
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32, copy=False)


def augmentation_methods() -> list[str]:
    return ["noise", "subsample", "scale", "crop", "jitter"]
