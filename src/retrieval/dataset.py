"""Dataset utilities for video-text embedding alignment."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.keypoints.augmentation import augment_sequence


KEYPOINT_DIM = 1662


@dataclass(frozen=True)
class VideoTextAlignmentSample:
    uid: str
    text: str
    keypoints: torch.Tensor
    text_embedding: torch.Tensor
    embedding_id: int
    source_row: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_embedding_index(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in read_csv(path):
        source_row = row.get("source_row", "").strip()
        embedding_id = row.get("embedding_id", "").strip()
        if source_row and embedding_id:
            mapping[int(source_row)] = int(embedding_id)
    return mapping


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


class VideoTextAlignmentDataset(torch.utils.data.Dataset[VideoTextAlignmentSample]):
    def __init__(
        self,
        manifest: Path,
        text_embeddings: np.ndarray,
        embedding_index: Path,
        text_column: str = "canonical_text",
        keypoint_column: str = "keypoint_path",
        max_frames: int = 512,
        sample_mode: str = "uniform",
        limit: int | None = None,
        augment: bool = False,
        augment_probability: float = 0.75,
        augment_methods: list[str] | None = None,
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

        self.text_embeddings = text_embeddings
        self.embedding_by_source_row = read_embedding_index(embedding_index)
        self.text_column = text_column
        self.keypoint_column = keypoint_column
        self.max_frames = max_frames
        self.sample_mode = sample_mode
        self.augment = augment
        self.augment_probability = augment_probability
        self.augment_methods = augment_methods or []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> VideoTextAlignmentSample:
        row = self.rows[index]
        source_row = int(row["_source_row"])
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
        embedding_id = self.embedding_by_source_row[source_row]
        text_embedding = self.text_embeddings[embedding_id].astype(np.float32, copy=True)
        np.nan_to_num(text_embedding, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return VideoTextAlignmentSample(
            uid=row.get("uid", ""),
            text=row[self.text_column],
            keypoints=torch.from_numpy(keypoints.copy()),
            text_embedding=torch.from_numpy(text_embedding),
            embedding_id=embedding_id,
            source_row=source_row,
        )


def collate_video_text_alignment(
    batch: list[VideoTextAlignmentSample],
) -> dict[str, object]:
    lengths = torch.tensor([item.keypoints.size(0) for item in batch], dtype=torch.long)
    keypoints = pad_sequence([item.keypoints for item in batch], batch_first=True)
    text_embeddings = torch.stack([item.text_embedding for item in batch]).float()
    return {
        "uids": [item.uid for item in batch],
        "texts": [item.text for item in batch],
        "source_rows": torch.tensor([item.source_row for item in batch], dtype=torch.long),
        "embedding_ids": torch.tensor([item.embedding_id for item in batch], dtype=torch.long),
        "keypoints": keypoints,
        "lengths": lengths,
        "text_embeddings": text_embeddings,
    }


@dataclass(frozen=True)
class RetrievalSample:
    uid: str
    text: str
    keypoints: torch.Tensor
    source_row: int


class RetrievalDataset(torch.utils.data.Dataset[RetrievalSample]):
    """End-to-end retrieval dataset: yields keypoints and the raw caption.

    Unlike :class:`VideoTextAlignmentDataset`, no precomputed text embeddings are
    needed - the caption text is tokenized and encoded by a trainable text
    encoder during training.
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

        self.text_column = text_column
        self.keypoint_column = keypoint_column
        self.max_frames = max_frames
        self.sample_mode = sample_mode
        self.augment = augment
        self.augment_probability = augment_probability
        self.augment_methods = augment_methods or []

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

        return RetrievalSample(
            uid=row.get("uid", ""),
            text=row[self.text_column],
            keypoints=torch.from_numpy(keypoints.copy()),
            source_row=int(row["_source_row"]),
        )


def collate_retrieval(batch: list[RetrievalSample]) -> dict[str, object]:
    lengths = torch.tensor([item.keypoints.size(0) for item in batch], dtype=torch.long)
    keypoints = pad_sequence([item.keypoints for item in batch], batch_first=True)
    return {
        "uids": [item.uid for item in batch],
        "texts": [item.text for item in batch],
        "source_rows": torch.tensor([item.source_row for item in batch], dtype=torch.long),
        "keypoints": keypoints,
        "lengths": lengths,
    }
