"""Dataset utilities for SignPose2Text training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


KEYPOINT_DIM = 1662


@dataclass(frozen=True)
class SignPose2TextSample:
    uid: str
    keypoints: torch.Tensor
    text: str
    source_row: int


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def uniform_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if num_frames <= max_frames:
        return np.arange(num_frames)
    return np.linspace(0, num_frames - 1, max_frames).round().astype(np.int64)


def random_crop_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if num_frames <= max_frames:
        return np.arange(num_frames)
    start = np.random.randint(0, num_frames - max_frames + 1)
    return np.arange(start, start + max_frames)


class SignPose2TextDataset(torch.utils.data.Dataset[SignPose2TextSample]):
    def __init__(
        self,
        manifest: Path,
        text_column: str = "canonical_text",
        keypoint_column: str = "keypoint_path",
        max_frames: int = 512,
        sample_mode: str = "uniform",
        require_keypoints: bool = True,
        limit: int | None = None,
    ) -> None:
        rows = read_manifest(manifest)
        if limit is not None:
            rows = rows[:limit]

        self.rows: list[dict[str, str]] = []
        for row_index, row in enumerate(rows):
            text = row.get(text_column, "").strip()
            keypoint_path = row.get(keypoint_column, "").strip()
            if not text:
                continue
            if require_keypoints and not keypoint_path:
                continue
            next_row = dict(row)
            next_row["_source_row"] = str(row_index)
            self.rows.append(next_row)

        self.text_column = text_column
        self.keypoint_column = keypoint_column
        self.max_frames = max_frames
        self.sample_mode = sample_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> SignPose2TextSample:
        row = self.rows[index]
        keypoint_path = Path(row[self.keypoint_column])
        arr = np.load(keypoint_path)
        if arr.ndim != 2 or arr.shape[1] != KEYPOINT_DIM:
            raise ValueError(f"Bad keypoint shape for {keypoint_path}: {arr.shape}")
        arr = np.nan_to_num(arr, copy=False).astype(np.float32, copy=False)

        if self.sample_mode == "random":
            indices = random_crop_indices(arr.shape[0], self.max_frames)
        elif self.sample_mode == "uniform":
            indices = uniform_indices(arr.shape[0], self.max_frames)
        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")
        arr = arr[indices]

        return SignPose2TextSample(
            uid=row.get("uid", ""),
            keypoints=torch.from_numpy(arr.copy()),
            text=row[self.text_column],
            source_row=int(row.get("_source_row", index)),
        )


class SignPose2TextCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_target_tokens: int = 96,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_target_tokens = max_target_tokens

    def __call__(self, batch: list[SignPose2TextSample]) -> dict[str, Any]:
        keypoints = [item.keypoints for item in batch]
        lengths = torch.tensor([item.keypoints.size(0) for item in batch], dtype=torch.long)
        padded_keypoints = pad_sequence(keypoints, batch_first=True)
        texts = [item.text for item in batch]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_target_tokens,
            return_tensors="pt",
        )
        labels = encoded["input_ids"]
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        return {
            "uids": [item.uid for item in batch],
            "source_rows": torch.tensor([item.source_row for item in batch], dtype=torch.long),
            "keypoints": padded_keypoints,
            "lengths": lengths,
            "labels": labels,
            "decoder_attention_mask": encoded["attention_mask"],
            "texts": texts,
        }
