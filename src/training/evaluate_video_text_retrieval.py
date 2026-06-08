"""Evaluate video-text alignment checkpoints with full-set retrieval metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modeling.keypoint_conformer import KeypointConformerEncoder
from src.training.video_text_alignment_dataset import (
    VideoTextAlignmentDataset,
    collate_video_text_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-size", type=int, default=1024)
    return parser.parse_args()


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return dict(checkpoint.get("config") or {})


def load_model(checkpoint: dict[str, Any], device: torch.device) -> KeypointConformerEncoder:
    config = checkpoint_config(checkpoint)
    model = KeypointConformerEncoder(
        model_dim=int(config.get("model_dim", 256)),
        projection_dim=int(config.get("projection_dim", 384)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 4)),
        downsample_stride=int(config.get("downsample_stride", 4)),
        dropout=float(config.get("dropout", 0.1)),
        normalize_output=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def encode_videos(
    model: KeypointConformerEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    video_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            video = model(keypoints, lengths).detach().cpu()
            text = batch["text_embeddings"].detach().cpu()
            video_embeddings.append(video)
            text_embeddings.append(text)
            for uid, source_row, target in zip(
                batch["uids"],
                batch["source_rows"].tolist(),
                batch["texts"],
            ):
                rows.append({"uid": uid, "source_row": str(source_row), "target": target})
            if batch_index % 20 == 0:
                print(f"encoded rows={len(rows)}", flush=True)
    video_tensor = F.normalize(torch.cat(video_embeddings, dim=0).float(), dim=-1, eps=1e-6)
    text_tensor = F.normalize(torch.cat(text_embeddings, dim=0).float(), dim=-1, eps=1e-6)
    return video_tensor, text_tensor, rows


def rank_metrics(
    query: torch.Tensor,
    candidates: torch.Tensor,
    chunk_size: int,
) -> dict[str, float]:
    n = query.size(0)
    ranks: list[int] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores = query[start:end] @ candidates.T
        target_scores = scores[:, start:end].diag().unsqueeze(1)
        # Rank is 1 plus number of candidates with a higher score.
        rank = (scores > target_scores).sum(dim=1) + 1
        ranks.extend(rank.cpu().tolist())

    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    return {
        "r1": float((rank_tensor <= 1).float().mean().item()),
        "r5": float((rank_tensor <= 5).float().mean().item()),
        "r10": float((rank_tensor <= 10).float().mean().item()),
        "mean_rank": float(rank_tensor.mean().item()),
        "median_rank": float(rank_tensor.median().item()),
    }


def write_json(path: Path, metrics: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def write_error_sample(
    path: Path,
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    rows: list[dict[str, str]],
    limit: int = 200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = video_embeddings @ text_embeddings.T
    target_scores = scores.diag().unsqueeze(1)
    ranks = (scores > target_scores).sum(dim=1) + 1
    worst = torch.argsort(ranks, descending=True)[:limit].tolist()
    fields = ["uid", "source_row", "target", "rank", "top_prediction", "top_score", "target_score"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx in worst:
            top_idx = int(torch.argmax(scores[idx]).item())
            writer.writerow(
                {
                    "uid": rows[idx]["uid"],
                    "source_row": rows[idx]["source_row"],
                    "target": rows[idx]["target"],
                    "rank": str(int(ranks[idx].item())),
                    "top_prediction": rows[top_idx]["target"],
                    "top_score": f"{float(scores[idx, top_idx].item()):.6f}",
                    "target_score": f"{float(scores[idx, idx].item()):.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_config(checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(checkpoint, device)

    text_embeddings = np.load(args.text_embeddings, mmap_mode="r")
    dataset = VideoTextAlignmentDataset(
        manifest=args.manifest,
        text_embeddings=text_embeddings,
        embedding_index=args.embedding_index,
        text_column=args.text_column,
        keypoint_column=args.keypoint_column,
        max_frames=args.max_frames or int(config.get("max_frames", 512)),
        sample_mode="uniform",
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_video_text_alignment,
        pin_memory=device.type == "cuda",
    )
    video, text, rows = encode_videos(model, loader, device)
    v2t = rank_metrics(video, text, args.chunk_size)
    t2v = rank_metrics(text, video, args.chunk_size)
    metrics: dict[str, float | str] = {
        "checkpoint": str(args.checkpoint),
        "rows": float(len(rows)),
        **{f"v2t_{key}": value for key, value in v2t.items()},
        **{f"t2v_{key}": value for key, value in t2v.items()},
    }
    write_json(args.output_json, metrics)
    if args.error_output:
        write_error_sample(args.error_output, video, text, rows)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"metrics saved: {args.output_json}")
    if args.error_output:
        print(f"errors saved : {args.error_output}")


if __name__ == "__main__":
    main()
