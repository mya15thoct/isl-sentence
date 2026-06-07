from __future__ import annotations

import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


KEYPOINT_DIM = 1662


def read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_rows(
    rows: list[dict[str, str]],
    val_ratio: float,
    seed: int,
    group_key: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if val_ratio <= 0:
        return rows, []

    rng = random.Random(seed)
    if not group_key:
        shuffled = list(rows)
        rng.shuffle(shuffled)
        val_size = max(1, int(len(shuffled) * val_ratio))
        return shuffled[val_size:], shuffled[:val_size]

    groups: dict[str, list[dict[str, str]]] = {}
    for idx, row in enumerate(rows):
        group = (
            row.get(group_key, "").strip()
            or row.get("keypoint_path", "").strip()
            or row.get("uid", "").strip()
            or f"row-{idx}"
        )
        groups.setdefault(group, []).append(row)

    shuffled_groups = list(groups.values())
    rng.shuffle(shuffled_groups)

    target_val_size = max(1, int(len(rows) * val_ratio))
    val: list[dict[str, str]] = []
    train: list[dict[str, str]] = []
    for group_rows in shuffled_groups:
        if len(val) < target_val_size:
            val.extend(group_rows)
        else:
            train.extend(group_rows)

    return train, val


def filter_usable_rows(
    rows: list[dict[str, str]],
    check_files: bool = True,
) -> list[dict[str, str]]:
    usable: list[dict[str, str]] = []
    for row in rows:
        keypoint_path = row.get("keypoint_path", "").strip()
        if not keypoint_path:
            continue

        status = row.get("extract_status", "").strip()
        if status and status not in {"ok", "skipped_exists"}:
            continue

        try:
            if int(float(row.get("num_frames", "1") or 1)) <= 0:
                continue
        except ValueError:
            pass

        if check_files and not Path(keypoint_path).is_file():
            continue

        usable.append(row)
    return usable


def sample_frames(
    keypoints: np.ndarray,
    max_frames: int | None,
    sample_mode: str,
) -> np.ndarray:
    if max_frames is None or max_frames <= 0 or keypoints.shape[0] <= max_frames:
        return keypoints

    if sample_mode == "center":
        start = (keypoints.shape[0] - max_frames) // 2
        return keypoints[start : start + max_frames]

    if sample_mode == "random":
        start = random.randint(0, keypoints.shape[0] - max_frames)
        return keypoints[start : start + max_frames]

    indices = np.linspace(0, keypoints.shape[0] - 1, max_frames).round().astype(np.int64)
    return keypoints[indices]


class VideoTextEmbeddingDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        text_embeddings: np.ndarray,
        max_frames: int | None = 512,
        sample_mode: str = "uniform",
    ):
        self.rows = rows
        self.text_embeddings = text_embeddings
        self.max_frames = max_frames
        self.sample_mode = sample_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        keypoints = np.load(row["keypoint_path"]).astype(np.float32, copy=False)
        if keypoints.ndim != 2 or keypoints.shape[1] != KEYPOINT_DIM:
            raise ValueError(f"Invalid keypoint shape {keypoints.shape}: {row['keypoint_path']}")

        keypoints = sample_frames(keypoints, self.max_frames, self.sample_mode).astype(
            np.float32,
            copy=True,
        )
        np.nan_to_num(keypoints, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        embedding_id = int(row["embedding_id"])
        text_embedding = self.text_embeddings[embedding_id].astype(np.float32, copy=True)
        np.nan_to_num(text_embedding, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "uid": row.get("uid", ""),
            "text": row.get("text", ""),
            "relation_type": row.get("relation_type", ""),
            "keypoints": torch.from_numpy(keypoints),
            "text_embedding": torch.from_numpy(text_embedding),
            "relation_weight": float(row.get("relation_weight") or 1.0),
            "length": keypoints.shape[0],
        }


def collate_video_text(batch: list[dict[str, object]]) -> dict[str, object]:
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feature_dim = batch[0]["keypoints"].shape[1]

    keypoints = torch.zeros(len(batch), max_len, feature_dim, dtype=torch.float32)
    text_embeddings = torch.stack([item["text_embedding"] for item in batch]).float()
    relation_weights = torch.tensor(
        [float(item.get("relation_weight", 1.0)) for item in batch],
        dtype=torch.float32,
    )

    for i, item in enumerate(batch):
        x = item["keypoints"]
        keypoints[i, : x.shape[0]] = x

    return {
        "uid": [item["uid"] for item in batch],
        "text": [item["text"] for item in batch],
        "relation_type": [item.get("relation_type", "") for item in batch],
        "keypoints": keypoints,
        "lengths": lengths,
        "text_embeddings": text_embeddings,
        "relation_weights": relation_weights,
    }
