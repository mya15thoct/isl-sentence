from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.video_text_dataset import (
    VideoTextEmbeddingDataset,
    collate_video_text,
    filter_usable_rows,
    read_index_csv,
    split_rows,
)
from src.video.conformer_encoder import KeypointConformerEncoder


DEFAULTS = {
    "val_ratio": 0.02,
    "split_group_key": "",
    "seed": 42,
    "max_frames": 512,
    "sample_mode": "uniform",
    "model_dim": 256,
    "projection_dim": 384,
    "num_layers": 4,
    "num_heads": 4,
    "downsample_stride": 4,
    "dropout": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a video-text checkpoint on retrieval metrics.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-errors", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--max-matrix-rows", type=int, default=20000)

    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--split-group-key")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sample-mode", choices=["uniform", "center", "random"])
    parser.add_argument("--model-dim", type=int)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--downsample-stride", type=int)
    parser.add_argument("--dropout", type=float)
    return parser.parse_args()


def cfg_value(args: argparse.Namespace, config: dict[str, Any], name: str) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    if name in config and config[name] is not None:
        return config[name]
    return DEFAULTS[name]


def find_bad_tensors(state_dict: dict[str, torch.Tensor]) -> list[str]:
    bad: list[str] = []
    for name, tensor in state_dict.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            bad.append(name)
    return bad


def select_rows(
    rows: list[dict[str, str]],
    split: str,
    val_ratio: float,
    seed: int,
    group_key: str | None = None,
) -> list[dict[str, str]]:
    if split == "all":
        return rows

    train_rows, val_rows = split_rows(rows, val_ratio, seed, group_key=group_key)
    return val_rows if split == "val" else train_rows


def build_model(config: dict[str, Any], args: argparse.Namespace) -> KeypointConformerEncoder:
    return KeypointConformerEncoder(
        model_dim=int(cfg_value(args, config, "model_dim")),
        projection_dim=int(cfg_value(args, config, "projection_dim")),
        num_layers=int(cfg_value(args, config, "num_layers")),
        num_heads=int(cfg_value(args, config, "num_heads")),
        downsample_stride=int(cfg_value(args, config, "downsample_stride")),
        dropout=float(cfg_value(args, config, "dropout")),
    )


def encode_videos(
    model: KeypointConformerEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    video_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    uids: list[str] = []
    texts: list[str] = []

    model.eval()
    start = time()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            text_embeddings = batch["text_embeddings"].float()

            video_embeddings = model(keypoints, lengths).detach().cpu().float()
            video_embeddings = torch.nan_to_num(
                video_embeddings,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            text_embeddings = torch.nan_to_num(
                text_embeddings,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            video_chunks.append(video_embeddings)
            text_chunks.append(text_embeddings)
            uids.extend(batch["uid"])
            texts.extend(batch["text"])

            if step % 20 == 0:
                seen = sum(chunk.size(0) for chunk in video_chunks)
                elapsed = max(time() - start, 1e-6)
                print(f"encoded rows={seen} batches={step} rows_per_sec={seen / elapsed:.1f}")

    return (
        torch.cat(video_chunks, dim=0),
        torch.cat(text_chunks, dim=0),
        uids,
        texts,
    )


def retrieval_metrics(logits: torch.Tensor) -> dict[str, float]:
    n = logits.size(0)
    labels = torch.arange(n)

    v2t_loss = F.cross_entropy(logits, labels)
    t2v_loss = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (v2t_loss + t2v_loss)

    diag = logits.diag()
    v2t_rank = (logits > diag[:, None]).sum(dim=1) + 1
    t2v_rank = (logits.t() > diag[:, None]).sum(dim=1) + 1

    metrics: dict[str, float] = {
        "loss": float(loss.item()),
        "v2t_loss": float(v2t_loss.item()),
        "t2v_loss": float(t2v_loss.item()),
        "rows": float(n),
    }
    for name, ranks in [("v2t", v2t_rank), ("t2v", t2v_rank)]:
        ranks_f = ranks.float()
        metrics[f"{name}_r1"] = float((ranks <= 1).float().mean().item())
        metrics[f"{name}_r5"] = float((ranks <= 5).float().mean().item())
        metrics[f"{name}_r10"] = float((ranks <= 10).float().mean().item())
        metrics[f"{name}_median_rank"] = float(ranks_f.median().item())
        metrics[f"{name}_mean_rank"] = float(ranks_f.mean().item())
    return metrics


def write_errors(
    path: Path,
    logits: torch.Tensor,
    uids: list[str],
    texts: list[str],
    max_rows: int = 200,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_scores, top_indices = logits.topk(k=min(5, logits.size(1)), dim=1)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_uid",
                "query_text",
                "true_rank",
                "top1_uid",
                "top1_text",
                "top1_score",
                "top5_uids",
                "top5_texts",
            ],
        )
        writer.writeheader()

        written = 0
        diag = logits.diag()
        ranks = (logits > diag[:, None]).sum(dim=1) + 1
        for i in torch.argsort(ranks, descending=True).tolist():
            if ranks[i].item() == 1:
                continue
            preds = top_indices[i].tolist()
            writer.writerow(
                {
                    "query_uid": uids[i],
                    "query_text": texts[i],
                    "true_rank": int(ranks[i].item()),
                    "top1_uid": uids[preds[0]],
                    "top1_text": texts[preds[0]],
                    "top1_score": float(top_scores[i, 0].item()),
                    "top5_uids": " | ".join(uids[j] for j in preds),
                    "top5_texts": " | ".join(texts[j] for j in preds),
                }
            )
            written += 1
            if written >= max_rows:
                break


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {}) or {}
    state_dict = checkpoint["model_state"]
    bad_tensors = find_bad_tensors(state_dict)

    val_ratio = float(cfg_value(args, config, "val_ratio"))
    split_group_key = str(cfg_value(args, config, "split_group_key") or "")
    seed = int(cfg_value(args, config, "seed"))
    max_frames = int(cfg_value(args, config, "max_frames"))
    sample_mode = str(cfg_value(args, config, "sample_mode"))
    eval_sample_mode = "center" if sample_mode == "random" else sample_mode

    rows = read_index_csv(args.index_csv)
    rows = filter_usable_rows(rows, check_files=not args.skip_file_check)
    eval_rows = select_rows(
        rows,
        args.split,
        val_ratio,
        seed,
        group_key=split_group_key or None,
    )
    if not eval_rows:
        raise SystemExit("No rows selected for evaluation.")
    if len(eval_rows) > args.max_matrix_rows:
        raise SystemExit(
            f"Selected {len(eval_rows)} rows, but max_matrix_rows is "
            f"{args.max_matrix_rows}. Evaluate val split or raise the limit."
        )

    text_embeddings = np.load(args.text_embeddings, mmap_mode="r")
    projection_dim = int(cfg_value(args, config, "projection_dim"))
    if text_embeddings.shape[1] != projection_dim:
        raise SystemExit(
            f"Text embedding dim is {text_embeddings.shape[1]}, "
            f"but projection_dim is {projection_dim}."
        )

    dataset = VideoTextEmbeddingDataset(
        eval_rows,
        text_embeddings,
        max_frames=max_frames,
        sample_mode=eval_sample_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_video_text,
    )

    device = torch.device(args.device)
    model = build_model(config, args).to(device)
    load_result = model.load_state_dict(state_dict, strict=False)

    print(f"checkpoint : {args.checkpoint}")
    print(f"epoch      : {checkpoint.get('epoch')}")
    print(f"split      : {args.split}")
    if split_group_key:
        print(f"split group: {split_group_key}")
    print(f"rows       : {len(dataset)}")
    print(f"bad tensors: {len(bad_tensors)}")
    if bad_tensors:
        print(f"bad sample : {bad_tensors[:10]}")
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"missing    : {load_result.missing_keys[:10]}")
        print(f"unexpected : {load_result.unexpected_keys[:10]}")

    video_embeddings, text_eval_embeddings, uids, texts = encode_videos(model, loader, device)
    video_embeddings = F.normalize(video_embeddings, dim=-1, eps=1e-6)
    text_eval_embeddings = F.normalize(text_eval_embeddings, dim=-1, eps=1e-6)

    scale = checkpoint.get("logit_scale", torch.tensor(1.0))
    scale_value = float(torch.as_tensor(scale).float().exp().clamp(max=100.0).item())
    logits = scale_value * video_embeddings @ text_eval_embeddings.t()
    metrics = retrieval_metrics(logits)
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "epoch": float(checkpoint.get("epoch", -1)),
            "split": args.split,
            "split_group_key": split_group_key,
            "bad_tensor_count": float(len(bad_tensors)),
            "logit_scale": scale_value,
            "video_finite": float(torch.isfinite(video_embeddings).all().item()),
            "text_finite": float(torch.isfinite(text_eval_embeddings).all().item()),
        }
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"metrics saved: {args.output_json}")

    if args.output_errors is not None:
        write_errors(args.output_errors, logits, uids, texts)
        print(f"errors saved : {args.output_errors}")


if __name__ == "__main__":
    main()
