"""Dataset utilities for end-to-end pose-text retrieval."""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.keypoints.augmentation import augment_sequence


KEYPOINT_DIM = 1662


def normalize_for_grouping(text: str) -> str:
    """Canonical key for caption-redundancy grouping.

    Captions that normalize to the same key (e.g. the many identical
    "let me tell you about it" rows in iSign) are treated as one group so that
    they are not used as false negatives of each other.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# MediaPipe FaceMesh landmark indices that carry sign-relevant non-manual markers
# (lips, eyebrows, eyes). All other face landmarks (contour, cheeks, forehead,
# nose bridge) are treated as noise and zeroed when ``face_keep`` is on.
FACE_KEEP_LANDMARKS = frozenset({
    # lips (outer + inner)
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178,
    181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321,
    324, 375, 402, 405, 409, 415,
    # eyebrows (left + right)
    46, 52, 53, 55, 63, 65, 66, 70, 105, 107,
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336,
    # eyes (left + right)
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 398, 466,
})


def _build_face_keep_mask() -> np.ndarray:
    """1/0 mask over the 1662-d frame that zeroes non-kept face landmarks."""
    mask = np.ones(KEYPOINT_DIM, dtype=np.float32)
    face_start = 132  # POSE_DIM; face block = [132 : 132+1404], 468 landmarks × 3
    for landmark in range(468):
        if landmark not in FACE_KEEP_LANDMARKS:
            mask[face_start + landmark * 3 : face_start + landmark * 3 + 3] = 0.0
    return mask


FACE_KEEP_MASK = _build_face_keep_mask()


def add_motion_features(keypoints: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Append per-frame velocity (Δ) and acceleration (ΔΔ) to raw positions.

    Returns ``(T, 3*KEYPOINT_DIM)`` = ``[position ; velocity ; acceleration]``.
    Velocity is the signed first difference (direction of motion, the meaningful
    part), acceleration the second difference (start/stop, direction change).
    Frame 0 velocity/acceleration are zero (no prior frame).

    When ``normalize`` (default), velocity and acceleration are each divided by
    their own per-sequence standard deviation. This (a) removes the absolute
    signing-speed scale, which is signer/frame-rate dependent, while keeping the
    sign/direction and the relative motion pattern, and (b) brings the motion
    streams to ~unit scale so they are comparable to the (shoulder-normalized)
    positions and are not drowned out inside the shared part-MLP. A single scalar
    std per stream is used so that "which keypoint moves more" is preserved.
    """
    velocity = np.zeros_like(keypoints)
    velocity[1:] = keypoints[1:] - keypoints[:-1]
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    if normalize:
        velocity = velocity / (float(velocity.std()) + 1e-6)
        acceleration = acceleration / (float(acceleration.std()) + 1e-6)
    return np.concatenate([keypoints, velocity, acceleration], axis=1).astype(np.float32)


def sample_keypoints(keypoints: np.ndarray, max_frames: int, sample_mode: str) -> np.ndarray:
    if max_frames <= 0 or keypoints.shape[0] <= max_frames:
        return keypoints
    if sample_mode == "random":
        start = random.randint(0, keypoints.shape[0] - max_frames)
        return keypoints[start : start + max_frames]
    if sample_mode == "center":
        start = (keypoints.shape[0] - max_frames) // 2
        return keypoints[start : start + max_frames]
    if sample_mode == "uniform":
        indices = np.linspace(0, keypoints.shape[0] - 1, max_frames).round().astype(np.int64)
        return keypoints[indices]
    raise ValueError(f"Unknown sample_mode: {sample_mode}")


@dataclass(frozen=True)
class RetrievalSample:
    uid: str
    text: str
    keypoints: torch.Tensor
    source_row: int
    group_id: int


class RetrievalDataset(torch.utils.data.Dataset[RetrievalSample]):
    """End-to-end retrieval dataset: yields keypoints and the raw caption.

    The caption text is tokenized and encoded by a trainable text encoder
    during training (no precomputed text embeddings).
    """

    def __init__(
        self,
        manifest: Path,
        text_column: str = "canonical_text",
        keypoint_column: str = "keypoint_path",
        max_frames: int = 512,
        sample_mode: str = "uniform",
        limit: int | None = None,
        augment: bool = False,
        augment_probability: float = 0.75,
        augment_methods: list[str] | None = None,
        motion_features: bool = False,
        face_keep: bool = False,
    ) -> None:
        rows = read_csv(manifest)
        if limit is not None:
            rows = rows[:limit]

        self.rows: list[dict[str, str]] = []
        for source_row, row in enumerate(rows):
            text = row.get(text_column, "").strip()
            keypoint_path = row.get(keypoint_column, "").strip()
            if not text or not keypoint_path:
                continue
            next_row = dict(row)
            next_row["_source_row"] = str(source_row)
            self.rows.append(next_row)

        # Redundancy grouping: identical normalized captions share a group id.
        group_map: dict[str, int] = {}
        self.group_ids: list[int] = []
        for row in self.rows:
            key = normalize_for_grouping(row[text_column])
            self.group_ids.append(group_map.setdefault(key, len(group_map)))
        self.num_groups = len(group_map)

        self.text_column = text_column
        self.keypoint_column = keypoint_column
        self.max_frames = max_frames
        self.sample_mode = sample_mode
        self.augment = augment
        self.augment_probability = augment_probability
        self.augment_methods = augment_methods or []
        self.motion_features = motion_features
        self.face_keep = face_keep

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RetrievalSample:
        row = self.rows[index]
        keypoint_path = Path(row[self.keypoint_column])
        keypoints = np.load(keypoint_path)
        if keypoints.ndim != 2 or keypoints.shape[1] != KEYPOINT_DIM:
            raise ValueError(f"Bad keypoint shape for {keypoint_path}: {keypoints.shape}")

        keypoints = np.nan_to_num(keypoints, copy=False).astype(np.float32, copy=False)
        if self.augment:
            keypoints = augment_sequence(
                keypoints,
                methods=self.augment_methods,
                probability=self.augment_probability,
            )
        keypoints = sample_keypoints(keypoints, self.max_frames, self.sample_mode)
        if self.face_keep:
            keypoints = keypoints * FACE_KEEP_MASK  # zero non-mouth/eyebrow/eye face points
        if self.motion_features:
            keypoints = add_motion_features(keypoints)

        return RetrievalSample(
            uid=row.get("uid", ""),
            text=row[self.text_column],
            keypoints=torch.from_numpy(keypoints.copy()),
            source_row=int(row["_source_row"]),
            group_id=self.group_ids[index],
        )


def collate_retrieval(batch: list[RetrievalSample]) -> dict[str, object]:
    lengths = torch.tensor([item.keypoints.size(0) for item in batch], dtype=torch.long)
    keypoints = pad_sequence([item.keypoints for item in batch], batch_first=True)
    return {
        "uids": [item.uid for item in batch],
        "texts": [item.text for item in batch],
        "source_rows": torch.tensor([item.source_row for item in batch], dtype=torch.long),
        "group_ids": torch.tensor([item.group_id for item in batch], dtype=torch.long),
        "keypoints": keypoints,
        "lengths": lengths,
    }
