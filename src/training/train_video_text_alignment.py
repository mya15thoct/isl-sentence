"""Train keypoint video embeddings to align with fixed text embeddings."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    video_embeddings = F.normalize(video_embeddings.float(), dim=-1, eps=1e-6)
    text_embeddings = F.normalize(text_embeddings.float(), dim=-1, eps=1e-6)
    logits = video_embeddings @ text_embeddings.T
    logits = logits / max(temperature, 1e-6)
    labels = torch.arange(logits.size(0), device=logits.device)
    v2t_loss = F.cross_entropy(logits, labels)
    t2v_loss = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (v2t_loss + t2v_loss)

    with torch.no_grad():
        v2t_acc = (logits.argmax(dim=1) == labels).float().mean().item()
        t2v_acc = (logits.argmax(dim=0) == labels).float().mean().item()
    return loss, {
        "v2t_acc": v2t_acc,
        "t2v_acc": t2v_acc,
        "v2t_loss": float(v2t_loss.detach().cpu()),
        "t2v_loss": float(t2v_loss.detach().cpu()),
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
    for key in ("keypoints", "lengths", "text_embeddings"):
        output[key] = batch[key].to(device, non_blocking=True)  # type: ignore[union-attr]
    return output


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "v2t_acc": 0.0, "t2v_acc": 0.0}
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
            )

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

        if training and args.print_every and step % args.print_every == 0:
            print(
                f"step={step} rows={rows_seen} "
                f"loss={totals['loss'] / rows_seen:.4f} "
                f"v2t={totals['v2t_acc'] / rows_seen:.3f} "
                f"t2v={totals['t2v_acc'] / rows_seen:.3f}",
                flush=True,
            )
        if not training and args.eval_max_batches > 0 and step >= args.eval_max_batches:
            break

    return {
        "loss": totals["loss"] / max(rows_seen, 1),
        "v2t_acc": totals["v2t_acc"] / max(rows_seen, 1),
        "t2v_acc": totals["t2v_acc"] / max(rows_seen, 1),
        "rows": float(rows_seen),
        "seconds": time.time() - start,
    }


def build_dataset(
    manifest: Path,
    embeddings: Path,
    index: Path,
    args: argparse.Namespace,
    sample_mode: str,
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
    )
    val_dataset = None
    if args.val_manifest is not None:
        val_dataset = build_dataset(
            args.val_manifest,
            args.val_text_embeddings or args.text_embeddings,
            args.val_embedding_index or args.embedding_index,
            args,
            sample_mode="uniform",
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
    print(f"save dir   : {args.save_dir}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(model, train_loader, device, args, optimizer, scaler)
        metrics: dict[str, Any] = {"train": train_metrics}
        print(
            "train "
            f"loss={train_metrics['loss']:.4f} "
            f"v2t={train_metrics['v2t_acc']:.3f} "
            f"t2v={train_metrics['t2v_acc']:.3f}",
            flush=True,
        )

        val_loss = train_metrics["loss"]
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, device, args, None, None)
            metrics["val"] = val_metrics
            val_loss = val_metrics["loss"]
            print(
                "val   "
                f"loss={val_metrics['loss']:.4f} "
                f"v2t={val_metrics['v2t_acc']:.3f} "
                f"t2v={val_metrics['t2v_acc']:.3f}",
                flush=True,
            )

        save_checkpoint(args.save_dir / "checkpoint_last.pt", model, optimizer, epoch, metrics, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.save_dir / "checkpoint_best.pt", model, optimizer, epoch, metrics, args)
            print(f"new best val loss: {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
