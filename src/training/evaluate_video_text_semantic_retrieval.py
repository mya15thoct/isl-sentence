"""Evaluate semantic quality of video-text retrieval predictions."""

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
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--semantic-thresholds",
        type=float,
        nargs="+",
        default=[0.70, 0.75, 0.80, 0.85],
        help="Text-text cosine thresholds used for semantic hit@k.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
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


def topk_indices(
    query: torch.Tensor,
    candidates: torch.Tensor,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    top_scores: list[torch.Tensor] = []
    top_indices: list[torch.Tensor] = []
    for start in range(0, query.size(0), chunk_size):
        end = min(start + chunk_size, query.size(0))
        scores = query[start:end] @ candidates.T
        values, indices = torch.topk(scores, k=min(k, candidates.size(0)), dim=1)
        top_scores.append(values.cpu())
        top_indices.append(indices.cpu())
    return torch.cat(top_scores, dim=0), torch.cat(top_indices, dim=0)


def summarize_semantic_retrieval(
    text_embeddings: torch.Tensor,
    top_indices: torch.Tensor,
    thresholds: list[float],
) -> dict[str, float]:
    target_indices = torch.arange(text_embeddings.size(0), dtype=torch.long)
    exact_hits = top_indices == target_indices.unsqueeze(1)
    gathered = text_embeddings[top_indices]
    target = text_embeddings.unsqueeze(1)
    semantic_scores = torch.sum(gathered * target, dim=-1)
    best_scores = semantic_scores.max(dim=1).values
    top1_scores = semantic_scores[:, 0]
    k5 = min(5, top_indices.size(1))
    k10 = min(10, top_indices.size(1))

    metrics: dict[str, float] = {
        "rows": float(text_embeddings.size(0)),
        "top_k": float(top_indices.size(1)),
        "exact_r1": float(exact_hits[:, :1].any(dim=1).float().mean().item()),
        "exact_r5": float(exact_hits[:, :k5].any(dim=1).float().mean().item()),
        "exact_r10": float(exact_hits[:, :k10].any(dim=1).float().mean().item()),
        "semantic_top1_mean": float(top1_scores.mean().item()),
        "semantic_top1_median": float(top1_scores.median().item()),
        "semantic_best_mean": float(best_scores.mean().item()),
        "semantic_best_median": float(best_scores.median().item()),
    }
    for threshold in thresholds:
        safe = str(threshold).replace(".", "p")
        metrics[f"semantic_r1_at_{safe}"] = float(
            (semantic_scores[:, :1] >= threshold).any(dim=1).float().mean().item()
        )
        metrics[f"semantic_r5_at_{safe}"] = float(
            (semantic_scores[:, :k5] >= threshold)
            .any(dim=1)
            .float()
            .mean()
            .item()
        )
        metrics[f"semantic_r10_at_{safe}"] = float(
            (semantic_scores[:, :k10] >= threshold).any(dim=1).float().mean().item()
        )
    return metrics


def write_json(path: Path, metrics: dict[str, float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def write_prediction_csv(
    path: Path,
    rows: list[dict[str, str]],
    text_embeddings: torch.Tensor,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "uid",
        "source_row",
        "target",
        "rank1_prediction",
        "rank1_retrieval_score",
        "rank1_semantic_score",
        "best_topk_semantic_score",
        "best_topk_semantic_rank",
        "topk_predictions",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_index, row in enumerate(rows):
            candidate_indices = top_indices[row_index]
            target = text_embeddings[row_index]
            candidate_text = text_embeddings[candidate_indices]
            semantic_scores = torch.sum(candidate_text * target.unsqueeze(0), dim=-1)
            best_rank = int(torch.argmax(semantic_scores).item())
            top_parts: list[str] = []
            for rank, candidate_index in enumerate(candidate_indices.tolist(), start=1):
                candidate = rows[candidate_index]["target"]
                sem = float(semantic_scores[rank - 1].item())
                score = float(top_scores[row_index, rank - 1].item())
                top_parts.append(f"{rank}:{score:.4f}:{sem:.4f}:{candidate[:100]}")
            writer.writerow(
                {
                    "uid": row["uid"],
                    "source_row": row["source_row"],
                    "target": row["target"],
                    "rank1_prediction": rows[int(candidate_indices[0].item())]["target"],
                    "rank1_retrieval_score": f"{float(top_scores[row_index, 0].item()):.6f}",
                    "rank1_semantic_score": f"{float(semantic_scores[0].item()):.6f}",
                    "best_topk_semantic_score": f"{float(semantic_scores[best_rank].item()):.6f}",
                    "best_topk_semantic_rank": str(best_rank + 1),
                    "topk_predictions": " | ".join(top_parts),
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
    top_scores, top_indices = topk_indices(video, text, args.top_k, args.chunk_size)
    metrics = summarize_semantic_retrieval(text, top_indices, args.semantic_thresholds)
    metrics = {
        "checkpoint": str(args.checkpoint),
        **metrics,
    }
    write_json(args.output_json, metrics)
    if args.output_csv:
        write_prediction_csv(args.output_csv, rows, text, top_scores, top_indices)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"metrics saved: {args.output_json}")
    if args.output_csv:
        print(f"predictions saved: {args.output_csv}")


if __name__ == "__main__":
    main()
