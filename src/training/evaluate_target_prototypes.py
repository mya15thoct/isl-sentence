from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_target_sentence_classifier import (
    TargetSentenceDataset,
    build_label_names,
    collate_target,
    read_manifest,
)
from src.video.conformer_encoder import KeypointConformerEncoder


DEFAULTS = {
    "model_dim": 256,
    "projection_dim": 384,
    "num_layers": 4,
    "num_heads": 4,
    "downsample_stride": 4,
    "dropout": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate target sentence recognition with train-set prototypes.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--errors-csv", type=Path)
    parser.add_argument("--class-text-embeddings", type=Path)
    parser.add_argument("--text-weight", type=float, default=0.0)
    parser.add_argument("--text-logit-scale", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--sample-mode",
        choices=["uniform", "center", "random"],
        default="uniform",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    return config.get(name, DEFAULTS[name])


def make_encoder(args: argparse.Namespace, checkpoint: dict[str, Any], device: torch.device) -> KeypointConformerEncoder:
    config = checkpoint.get("config", {}) or {}
    encoder = KeypointConformerEncoder(
        model_dim=int(cfg_value(args, config, "model_dim")),
        projection_dim=int(cfg_value(args, config, "projection_dim")),
        num_layers=int(cfg_value(args, config, "num_layers")),
        num_heads=int(cfg_value(args, config, "num_heads")),
        downsample_stride=int(cfg_value(args, config, "downsample_stride")),
        dropout=float(cfg_value(args, config, "dropout")),
    ).to(device)
    state = checkpoint.get("encoder_state") or checkpoint.get("model_state")
    result = encoder.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        print(
            "warning: encoder key mismatch "
            f"missing={result.missing_keys[:10]} "
            f"unexpected={result.unexpected_keys[:10]}"
        )
    encoder.eval()
    return encoder


def make_loader(rows: list[dict[str, str]], args: argparse.Namespace) -> DataLoader:
    config_max_frames = args.max_frames if args.max_frames is not None else 512
    dataset = TargetSentenceDataset(
        rows,
        max_frames=config_max_frames,
        sample_mode=args.sample_mode,
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_target,
    )


def encode_rows(
    encoder: KeypointConformerEncoder,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    loader = make_loader(rows, args)
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    uids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            emb = encoder(keypoints, lengths).detach().cpu().float()
            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(emb)
            labels.append(batch["labels"].cpu())
            uids.extend(batch["uid"])
    return torch.cat(embeddings), torch.cat(labels), uids


def build_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    embeddings = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
    prototypes = torch.zeros(num_classes, embeddings.size(1), dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for emb, label in zip(embeddings, labels):
        prototypes[int(label)] += emb
        counts[int(label)] += 1
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise SystemExit(f"Cannot build prototypes; missing train classes: {missing[:20]}")
    prototypes = prototypes / counts[:, None]
    return F.normalize(prototypes, dim=-1, eps=1e-6)


def load_class_text_embeddings(
    path: Path | None,
    num_classes: int,
    embedding_dim: int,
) -> torch.Tensor | None:
    if path is None:
        return None
    embeddings = np.load(path).astype(np.float32, copy=True)
    if embeddings.shape != (num_classes, embedding_dim):
        raise SystemExit(
            "Class text embedding shape mismatch: "
            f"expected {(num_classes, embedding_dim)}, got {embeddings.shape}"
        )
    np.nan_to_num(embeddings, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(torch.from_numpy(embeddings).float(), dim=-1, eps=1e-6)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    top1 = logits.argmax(dim=1)
    top5 = logits.topk(min(5, logits.size(1)), dim=1).indices
    return {
        "top1": float((top1 == labels).float().mean().item()),
        "top5": float((top5 == labels[:, None]).any(dim=1).float().mean().item()),
        "rows": float(labels.numel()),
        "mean_rank": mean_rank(logits, labels),
        "median_rank": median_rank(logits, labels),
    }


def ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    order = logits.argsort(dim=1, descending=True)
    return (order == labels[:, None]).nonzero()[:, 1] + 1


def mean_rank(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(ranks(logits, labels).float().mean().item())


def median_rank(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(ranks(logits, labels).float().median().item())


def evaluate_split(
    prototypes: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_text_embeddings: torch.Tensor | None = None,
    text_weight: float = 0.0,
    text_logit_scale: float = 20.0,
) -> tuple[dict[str, float], torch.Tensor]:
    embeddings = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
    logits = embeddings @ prototypes.t()
    if class_text_embeddings is not None and text_weight > 0:
        logits = logits + text_weight * text_logit_scale * (embeddings @ class_text_embeddings.t())
    return metrics_from_logits(logits, labels), logits


def write_errors(
    path: Path,
    rows: list[dict[str, str]],
    logits: torch.Tensor,
    label_names: list[str],
    split: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    topk = logits.topk(min(5, logits.size(1)), dim=1)
    fields = [
        "split",
        "uid",
        "true_label_id",
        "true_label",
        "pred_label_id",
        "pred_label",
        "top5",
        "correct",
        "sentence",
        "gloss",
        "keypoint_path",
    ]
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for row, pred_ids in zip(rows, topk.indices.tolist()):
            true_id = int(row["label_id"])
            pred_id = int(pred_ids[0])
            if pred_id == true_id:
                continue
            writer.writerow(
                {
                    "split": split,
                    "uid": row.get("uid", ""),
                    "true_label_id": true_id,
                    "true_label": label_names[true_id],
                    "pred_label_id": pred_id,
                    "pred_label": label_names[pred_id],
                    "top5": " | ".join(label_names[int(idx)] for idx in pred_ids),
                    "correct": "0",
                    "sentence": row.get("sentence", ""),
                    "gloss": row.get("gloss", ""),
                    "keypoint_path": row.get("keypoint_path", ""),
                }
            )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = make_encoder(args, checkpoint, device)

    train_rows = read_manifest(args.train_csv)
    val_rows = read_manifest(args.val_csv)
    test_rows = read_manifest(args.test_csv) if args.test_csv else []
    label_names = checkpoint.get("label_names") or build_label_names(train_rows + val_rows + test_rows)
    num_classes = len(label_names)

    train_embeddings, train_labels, _ = encode_rows(encoder, train_rows, args, device)
    prototypes = build_prototypes(train_embeddings, train_labels, num_classes)
    class_text_embeddings = load_class_text_embeddings(
        args.class_text_embeddings,
        num_classes,
        train_embeddings.size(1),
    )

    val_embeddings, val_labels, _ = encode_rows(encoder, val_rows, args, device)
    val_metrics, val_logits = evaluate_split(
        prototypes,
        val_embeddings,
        val_labels,
        class_text_embeddings=class_text_embeddings,
        text_weight=args.text_weight,
        text_logit_scale=args.text_logit_scale,
    )
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "test_csv": str(args.test_csv) if args.test_csv else "",
        "class_text_embeddings": str(args.class_text_embeddings) if args.class_text_embeddings else "",
        "text_weight": args.text_weight,
        "num_classes": num_classes,
        "train_rows": len(train_rows),
        "val": val_metrics,
    }

    print(f"val  top1={val_metrics['top1']:.3f} top5={val_metrics['top5']:.3f}")

    if test_rows:
        test_embeddings, test_labels, _ = encode_rows(encoder, test_rows, args, device)
        test_metrics, test_logits = evaluate_split(
            prototypes,
            test_embeddings,
            test_labels,
            class_text_embeddings=class_text_embeddings,
            text_weight=args.text_weight,
            text_logit_scale=args.text_logit_scale,
        )
        result["test"] = test_metrics
        print(f"test top1={test_metrics['top1']:.3f} top5={test_metrics['top5']:.3f}")
    else:
        test_logits = None

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {args.output_json}")

    if args.errors_csv:
        if args.errors_csv.exists():
            args.errors_csv.unlink()
        write_errors(args.errors_csv, val_rows, val_logits, label_names, "val")
        if test_rows and test_logits is not None:
            write_errors(args.errors_csv, test_rows, test_logits, label_names, "test")
        print(f"errors: {args.errors_csv}")


if __name__ == "__main__":
    main()
