"""Train keypoint video embeddings to align with fixed text embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modeling.keypoint_conformer import KeypointConformerEncoder
from src.training.keypoint_augmentation import augmentation_methods
from src.training.video_text_alignment_dataset import (
    VideoTextAlignmentDataset,
    collate_video_text_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--embedding-index", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--val-text-embeddings", type=Path)
    parser.add_argument("--val-embedding-index", type=Path)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--sample-mode", choices=("uniform", "center", "random"), default="random")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-prob", type=float, default=0.75)
    parser.add_argument(
        "--augment-methods",
        nargs="+",
        choices=augmentation_methods(),
        default=["noise", "subsample", "scale", "crop", "jitter"],
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--semantic-positive-weight",
        type=float,
        default=0.0,
        help="Blend in semantic multi-positive loss based on text-text similarity.",
    )
    parser.add_argument(
        "--semantic-positive-threshold",
        type=float,
        default=0.82,
        help="Text cosine threshold for off-diagonal positives inside each batch.",
    )
    parser.add_argument(
        "--semantic-positive-top-k",
        type=int,
        default=0,
        help="Also keep the top-k most similar text items in the batch as soft positives.",
    )
    parser.add_argument(
        "--semantic-positive-power",
        type=float,
        default=1.0,
        help="Exponent applied to off-diagonal text similarity weights.",
    )
    parser.add_argument(
        "--semantic-graph",
        type=Path,
        help="Optional train semantic graph .npz from src/text/build_text_semantic_graph.py.",
    )
    parser.add_argument(
        "--val-semantic-graph",
        type=Path,
        help="Optional val semantic graph .npz. If omitted, val global loss uses exact positives.",
    )
    parser.add_argument(
        "--global-text-loss-weight",
        type=float,
        default=0.0,
        help="Blend batch loss with full text-bank loss. 0 keeps in-batch training.",
    )
    parser.add_argument(
        "--global-loss-warmup-epochs",
        type=float,
        default=0.0,
        help="Linearly ramp global text-bank loss over this many epochs.",
    )
    parser.add_argument(
        "--global-semantic-weight",
        type=float,
        default=0.5,
        help="Inside text-bank loss, blend exact CE with semantic multi-positive loss.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--downsample-stride", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


@dataclass
class AlignmentContext:
    text_bank: torch.Tensor
    positive_ids: torch.Tensor | None = None
    positive_weights: torch.Tensor | None = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
    semantic_positive_weight: float = 0.0,
    semantic_positive_threshold: float = 0.82,
    semantic_positive_top_k: int = 0,
    semantic_positive_power: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    video_embeddings = F.normalize(video_embeddings.float(), dim=-1, eps=1e-6)
    text_embeddings = F.normalize(text_embeddings.float(), dim=-1, eps=1e-6)
    logits = video_embeddings @ text_embeddings.T
    logits = logits / max(temperature, 1e-6)
    labels = torch.arange(logits.size(0), device=logits.device)
    hard_v2t_loss = F.cross_entropy(logits, labels)
    hard_t2v_loss = F.cross_entropy(logits.T, labels)
    hard_loss = 0.5 * (hard_v2t_loss + hard_t2v_loss)

    semantic_loss = torch.zeros((), device=logits.device)
    positive_pairs_per_row = 1.0
    if semantic_positive_weight > 0:
        positive_targets = semantic_positive_targets(
            text_embeddings,
            threshold=semantic_positive_threshold,
            top_k=semantic_positive_top_k,
            power=semantic_positive_power,
        )
        semantic_v2t_loss = soft_cross_entropy(logits, positive_targets)
        semantic_t2v_loss = soft_cross_entropy(logits.T, positive_targets)
        semantic_loss = 0.5 * (semantic_v2t_loss + semantic_t2v_loss)
        weight = min(max(float(semantic_positive_weight), 0.0), 1.0)
        loss = (1.0 - weight) * hard_loss + weight * semantic_loss
        with torch.no_grad():
            positive_pairs_per_row = float((positive_targets > 0).sum(dim=1).float().mean().item())
    else:
        loss = hard_loss

    with torch.no_grad():
        v2t_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        t2v_acc = (logits.argmax(dim=0) == labels).float().mean().item()
    return loss, {
        "v2t_acc": v2t_acc,
        "t2v_acc": t2v_acc,
        "hard_loss": float(hard_loss.detach().cpu()),
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "positive_pairs_per_row": positive_pairs_per_row,
    }


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def semantic_positive_targets(
    text_embeddings: torch.Tensor,
    threshold: float,
    top_k: int,
    power: float,
) -> torch.Tensor:
    similarity = text_embeddings @ text_embeddings.T
    batch_size = similarity.size(0)
    diagonal = torch.eye(batch_size, device=similarity.device, dtype=torch.bool)
    positive_mask = diagonal | (similarity >= threshold)

    if top_k > 0 and batch_size > 1:
        k = min(top_k + 1, batch_size)
        _, indices = torch.topk(similarity, k=k, dim=1)
        topk_mask = torch.zeros_like(positive_mask)
        topk_mask.scatter_(1, indices, True)
        positive_mask = positive_mask | topk_mask

    weights = torch.where(positive_mask, similarity.clamp_min(0.0), torch.zeros_like(similarity))
    if power != 1.0:
        weights = weights.pow(power)
    weights = torch.where(diagonal, torch.ones_like(weights), weights)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights


def global_text_bank_loss(
    video_embeddings: torch.Tensor,
    labels: torch.Tensor,
    context: AlignmentContext,
    temperature: float,
    semantic_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    video_embeddings = F.normalize(video_embeddings.float(), dim=-1, eps=1e-6)
    text_bank = F.normalize(context.text_bank.float(), dim=-1, eps=1e-6)
    logits = video_embeddings @ text_bank.T
    logits = logits / max(temperature, 1e-6)
    labels = labels.to(device=logits.device, dtype=torch.long)
    exact_loss = F.cross_entropy(logits, labels)

    semantic_loss = exact_loss
    positive_pairs_per_row = 1.0
    if context.positive_ids is not None and context.positive_weights is not None:
        positive_ids = context.positive_ids[labels]
        positive_weights = context.positive_weights[labels].to(logits.device)
        valid_mask = positive_ids >= 0
        safe_ids = positive_ids.clamp_min(0)
        positive_logits = logits.gather(1, safe_ids)
        positive_weights = positive_weights.masked_fill(~valid_mask, 0.0)
        positive_weights = positive_weights / positive_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_positive_logits = positive_logits + torch.log(positive_weights.clamp_min(1e-12))
        weighted_positive_logits = weighted_positive_logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        numerator = torch.logsumexp(weighted_positive_logits, dim=1)
        denominator = torch.logsumexp(logits, dim=1)
        semantic_loss = -(numerator - denominator).mean()
        with torch.no_grad():
            positive_pairs_per_row = float(valid_mask.float().sum(dim=1).mean().item())

    weight = min(max(float(semantic_weight), 0.0), 1.0)
    loss = (1.0 - weight) * exact_loss + weight * semantic_loss
    with torch.no_grad():
        top_k = min(10, logits.size(1))
        top_indices = torch.topk(logits, k=top_k, dim=1).indices
        hits = top_indices == labels.unsqueeze(1)
        v2t_acc = hits[:, :1].any(dim=1).float().mean().item()
        r5 = hits[:, : min(5, top_k)].any(dim=1).float().mean().item()
        r10 = hits[:, :top_k].any(dim=1).float().mean().item()
        target_scores = logits.gather(1, labels.unsqueeze(1))
        ranks = (logits > target_scores).sum(dim=1).float() + 1.0
        mean_rank = ranks.mean().item()
    return loss, {
        "global_exact_loss": float(exact_loss.detach().cpu()),
        "global_semantic_loss": float(semantic_loss.detach().cpu()),
        "global_v2t_acc": v2t_acc,
        "global_v2t_r5": r5,
        "global_v2t_r10": r10,
        "global_mean_rank": mean_rank,
        "global_positive_pairs_per_row": positive_pairs_per_row,
    }


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state(model),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    output = dict(batch)
    for key in ("keypoints", "lengths", "text_embeddings", "embedding_ids"):
        output[key] = batch[key].to(device, non_blocking=True)  # type: ignore[union-attr]
    return output


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    context: AlignmentContext | None = None,
    epoch: int = 1,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "v2t_acc": 0.0,
        "t2v_acc": 0.0,
        "hard_loss": 0.0,
        "semantic_loss": 0.0,
        "positive_pairs_per_row": 0.0,
        "global_loss": 0.0,
        "global_v2t_acc": 0.0,
        "global_v2t_r5": 0.0,
        "global_v2t_r10": 0.0,
        "global_mean_rank": 0.0,
        "global_positive_pairs_per_row": 0.0,
        "global_loss_weight": 0.0,
    }
    rows_seen = 0
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = bool(scaler is not None and scaler.is_enabled())
        with torch.cuda.amp.autocast(enabled=use_amp):
            video_embeddings = model(batch["keypoints"], batch["lengths"])
            loss, metrics = contrastive_loss(
                video_embeddings,
                batch["text_embeddings"],
                args.temperature,
                semantic_positive_weight=args.semantic_positive_weight,
                semantic_positive_threshold=args.semantic_positive_threshold,
                semantic_positive_top_k=args.semantic_positive_top_k,
                semantic_positive_power=args.semantic_positive_power,
            )
            if context is not None and args.global_text_loss_weight > 0:
                effective_global_weight = effective_global_loss_weight(
                    args.global_text_loss_weight,
                    args.global_loss_warmup_epochs,
                    epoch,
                )
                global_loss, global_metrics = global_text_bank_loss(
                    video_embeddings=video_embeddings,
                    labels=batch["embedding_ids"],
                    context=context,
                    temperature=args.temperature,
                    semantic_weight=args.global_semantic_weight,
                )
                loss = (1.0 - effective_global_weight) * loss + effective_global_weight * global_loss
                metrics.update(global_metrics)
                metrics["global_loss"] = float(global_loss.detach().cpu())
                metrics["global_loss_weight"] = effective_global_weight
            else:
                metrics["global_loss"] = 0.0
                metrics["global_v2t_acc"] = 0.0
                metrics["global_v2t_r5"] = 0.0
                metrics["global_v2t_r10"] = 0.0
                metrics["global_mean_rank"] = 0.0
                metrics["global_positive_pairs_per_row"] = 0.0
                metrics["global_loss_weight"] = 0.0

        if training:
            assert optimizer is not None
            assert scaler is not None
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        batch_size = int(batch["keypoints"].size(0))
        rows_seen += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["v2t_acc"] += metrics["v2t_acc"] * batch_size
        totals["t2v_acc"] += metrics["t2v_acc"] * batch_size
        totals["hard_loss"] += metrics["hard_loss"] * batch_size
        totals["semantic_loss"] += metrics["semantic_loss"] * batch_size
        totals["positive_pairs_per_row"] += metrics["positive_pairs_per_row"] * batch_size
        totals["global_loss"] += metrics["global_loss"] * batch_size
        totals["global_v2t_acc"] += metrics["global_v2t_acc"] * batch_size
        totals["global_v2t_r5"] += metrics["global_v2t_r5"] * batch_size
        totals["global_v2t_r10"] += metrics["global_v2t_r10"] * batch_size
        totals["global_mean_rank"] += metrics["global_mean_rank"] * batch_size
        totals["global_positive_pairs_per_row"] += metrics["global_positive_pairs_per_row"] * batch_size
        totals["global_loss_weight"] += metrics["global_loss_weight"] * batch_size

        if training and args.print_every and step % args.print_every == 0:
            print(
                f"step={step} rows={rows_seen} "
                f"loss={totals['loss'] / rows_seen:.4f} "
                f"v2t={totals['v2t_acc'] / rows_seen:.3f} "
                f"t2v={totals['t2v_acc'] / rows_seen:.3f} "
                f"pos={totals['positive_pairs_per_row'] / rows_seen:.2f} "
                f"gb_v2t={totals['global_v2t_acc'] / rows_seen:.3f} "
                f"gb_r10={totals['global_v2t_r10'] / rows_seen:.3f} "
                f"gb_rank={totals['global_mean_rank'] / rows_seen:.0f} "
                f"gb_w={totals['global_loss_weight'] / rows_seen:.2f} "
                f"gb_pos={totals['global_positive_pairs_per_row'] / rows_seen:.2f}",
                flush=True,
            )
        if not training and args.eval_max_batches > 0 and step >= args.eval_max_batches:
            break

    return {
        "loss": totals["loss"] / max(rows_seen, 1),
        "v2t_acc": totals["v2t_acc"] / max(rows_seen, 1),
        "t2v_acc": totals["t2v_acc"] / max(rows_seen, 1),
        "hard_loss": totals["hard_loss"] / max(rows_seen, 1),
        "semantic_loss": totals["semantic_loss"] / max(rows_seen, 1),
        "positive_pairs_per_row": totals["positive_pairs_per_row"] / max(rows_seen, 1),
        "global_loss": totals["global_loss"] / max(rows_seen, 1),
        "global_v2t_acc": totals["global_v2t_acc"] / max(rows_seen, 1),
        "global_v2t_r5": totals["global_v2t_r5"] / max(rows_seen, 1),
        "global_v2t_r10": totals["global_v2t_r10"] / max(rows_seen, 1),
        "global_mean_rank": totals["global_mean_rank"] / max(rows_seen, 1),
        "global_positive_pairs_per_row": totals["global_positive_pairs_per_row"] / max(rows_seen, 1),
        "global_loss_weight": totals["global_loss_weight"] / max(rows_seen, 1),
        "rows": float(rows_seen),
        "seconds": time.time() - start,
    }


def effective_global_loss_weight(
    target_weight: float,
    warmup_epochs: float,
    epoch: int,
) -> float:
    target = min(max(float(target_weight), 0.0), 1.0)
    if warmup_epochs <= 0:
        return target
    progress = min(max(float(epoch) / warmup_epochs, 0.0), 1.0)
    return target * progress


def build_dataset(
    manifest: Path,
    embeddings: Path,
    index: Path,
    args: argparse.Namespace,
    sample_mode: str,
    augment: bool,
) -> VideoTextAlignmentDataset:
    text_embeddings = np.load(embeddings, mmap_mode="r")
    if text_embeddings.shape[1] != args.projection_dim:
        raise SystemExit(
            f"Text embedding dim is {text_embeddings.shape[1]}, "
            f"but projection_dim is {args.projection_dim}."
        )
    return VideoTextAlignmentDataset(
        manifest=manifest,
        text_embeddings=text_embeddings,
        embedding_index=index,
        text_column=args.text_column,
        keypoint_column=args.keypoint_column,
        max_frames=args.max_frames,
        sample_mode=sample_mode,
        limit=args.limit,
        augment=augment,
        augment_probability=args.augment_prob,
        augment_methods=args.augment_methods,
    )


def load_alignment_context(
    embeddings: Path,
    graph_path: Path | None,
    args: argparse.Namespace,
    device: torch.device,
) -> AlignmentContext:
    text_embeddings = np.load(embeddings, mmap_mode="r")
    if text_embeddings.shape[1] != args.projection_dim:
        raise SystemExit(
            f"Text embedding dim is {text_embeddings.shape[1]}, "
            f"but projection_dim is {args.projection_dim}."
        )
    text_bank_array = np.asarray(text_embeddings, dtype=np.float32).copy()
    np.nan_to_num(text_bank_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    text_bank = torch.from_numpy(text_bank_array).to(device)
    text_bank = F.normalize(text_bank.float(), dim=-1, eps=1e-6)

    positive_ids = None
    positive_weights = None
    if graph_path is not None:
        graph = np.load(graph_path)
        ids = np.asarray(graph["positive_ids"], dtype=np.int64).copy()
        weights = np.asarray(graph["positive_weights"], dtype=np.float32).copy()
        if ids.shape != weights.shape:
            raise SystemExit(f"Bad semantic graph shapes: ids={ids.shape}, weights={weights.shape}.")
        if ids.shape[0] != text_bank.size(0):
            raise SystemExit(
                f"Semantic graph has {ids.shape[0]} rows, but text bank has {text_bank.size(0)} rows."
            )
        positive_ids = torch.from_numpy(ids).to(device)
        positive_weights = torch.from_numpy(weights).to(device)

    return AlignmentContext(
        text_bank=text_bank,
        positive_ids=positive_ids,
        positive_weights=positive_weights,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_dataset = build_dataset(
        args.manifest,
        args.text_embeddings,
        args.embedding_index,
        args,
        sample_mode=args.sample_mode,
        augment=args.augment,
    )
    val_dataset = None
    if args.val_manifest is not None:
        val_dataset = build_dataset(
            args.val_manifest,
            args.val_text_embeddings or args.text_embeddings,
            args.val_embedding_index or args.embedding_index,
            args,
            sample_mode="uniform",
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_video_text_alignment,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_video_text_alignment,
            pin_memory=device.type == "cuda",
        )

    train_context = None
    val_context = None
    if args.global_text_loss_weight > 0:
        train_context = load_alignment_context(args.text_embeddings, args.semantic_graph, args, device)
        if val_dataset is not None:
            val_context = load_alignment_context(
                args.val_text_embeddings or args.text_embeddings,
                args.val_semantic_graph,
                args,
                device,
            )

    model: nn.Module = KeypointConformerEncoder(
        model_dim=args.model_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        downsample_stride=args.downsample_stride,
        dropout=args.dropout,
        normalize_output=True,
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"train rows : {len(train_dataset)}")
    print(f"val rows   : {len(val_dataset) if val_dataset is not None else 0}")
    print(f"device     : {device}")
    print(f"gpus       : {torch.cuda.device_count() if device.type == 'cuda' else 0}")
    print(f"global text: {args.global_text_loss_weight > 0}")
    print(f"augment    : {args.augment} {args.augment_methods if args.augment else ''}")
    if train_context is not None:
        print(f"text bank  : {tuple(train_context.text_bank.shape)}")
        print(f"text graph : {args.semantic_graph or 'exact-only'}")
    print(f"save dir   : {args.save_dir}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer,
            scaler,
            train_context,
            epoch=epoch,
        )
        metrics: dict[str, Any] = {"train": train_metrics}
        print(
            "train "
            f"loss={train_metrics['loss']:.4f} "
            f"v2t={train_metrics['v2t_acc']:.3f} "
            f"t2v={train_metrics['t2v_acc']:.3f} "
            f"pos={train_metrics['positive_pairs_per_row']:.2f} "
            f"gb_v2t={train_metrics['global_v2t_acc']:.3f} "
            f"gb_r10={train_metrics['global_v2t_r10']:.3f} "
            f"gb_rank={train_metrics['global_mean_rank']:.0f} "
            f"gb_w={train_metrics['global_loss_weight']:.2f} "
            f"gb_pos={train_metrics['global_positive_pairs_per_row']:.2f}",
            flush=True,
        )

        val_loss = train_metrics["loss"]
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    args,
                    None,
                    None,
                    val_context,
                    epoch=epoch,
                )
            metrics["val"] = val_metrics
            val_loss = val_metrics["loss"]
            print(
                "val   "
                f"loss={val_metrics['loss']:.4f} "
                f"v2t={val_metrics['v2t_acc']:.3f} "
                f"t2v={val_metrics['t2v_acc']:.3f} "
                f"pos={val_metrics['positive_pairs_per_row']:.2f} "
                f"gb_v2t={val_metrics['global_v2t_acc']:.3f} "
                f"gb_r10={val_metrics['global_v2t_r10']:.3f} "
                f"gb_rank={val_metrics['global_mean_rank']:.0f} "
                f"gb_w={val_metrics['global_loss_weight']:.2f} "
                f"gb_pos={val_metrics['global_positive_pairs_per_row']:.2f}",
                flush=True,
            )

        save_checkpoint(args.save_dir / "checkpoint_last.pt", model, optimizer, epoch, metrics, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.save_dir / "checkpoint_best.pt", model, optimizer, epoch, metrics, args)
            print(f"new best val loss: {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
