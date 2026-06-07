"""Train a baseline SignPose2Text model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.modeling.signpose2text import ConformerT5SignPose2Text
from src.training.signpose2text_dataset import (
    SignPose2TextCollator,
    SignPose2TextDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--text-model", default="t5-small")
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--max-target-tokens", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--generate-every-epoch", action="store_true")
    parser.add_argument("--generate-batches", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--keypoint-model-dim", type=int, default=256)
    parser.add_argument("--keypoint-layers", type=int, default=4)
    parser.add_argument("--keypoint-heads", type=int, default=4)
    parser.add_argument("--downsample-stride", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val = sorted(indices[:val_size])
    train = sorted(indices[val_size:])
    return train, val


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for key in ("keypoints", "lengths", "labels", "decoder_attention_mask"):
        output[key] = batch[key].to(device, non_blocking=True)
    return output


def save_checkpoint(
    path: Path,
    model: ConformerT5SignPose2Text,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def train_one_epoch(
    model: ConformerT5SignPose2Text,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_rows = 0
    start = time.time()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                keypoints=batch["keypoints"],
                lengths=batch["lengths"],
                labels=batch["labels"],
                decoder_attention_mask=batch["decoder_attention_mask"],
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        rows = int(batch["keypoints"].size(0))
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows

        if args.print_every and step % args.print_every == 0:
            print(
                f"step={step} rows={total_rows} "
                f"loss={total_loss / max(total_rows, 1):.4f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(total_rows, 1),
        "rows": float(total_rows),
        "seconds": time.time() - start,
    }


@torch.no_grad()
def evaluate(
    model: ConformerT5SignPose2Text,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    samples: list[dict[str, str]] = []
    start = time.time()

    for batch_index, batch in enumerate(loader):
        raw_batch = batch
        batch = move_batch(batch, device)
        outputs = model(
            keypoints=batch["keypoints"],
            lengths=batch["lengths"],
            labels=batch["labels"],
            decoder_attention_mask=batch["decoder_attention_mask"],
        )
        rows = int(batch["keypoints"].size(0))
        total_loss += float(outputs.loss.detach().cpu()) * rows
        total_rows += rows

        if args.generate_every_epoch and batch_index < args.generate_batches:
            generated = model.generate(
                keypoints=batch["keypoints"],
                lengths=batch["lengths"],
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            predictions = model.decode(generated)
            for uid, target, pred in zip(raw_batch["uids"], raw_batch["texts"], predictions):
                samples.append({"uid": uid, "target": target, "prediction": pred})

    return {
        "loss": total_loss / max(total_rows, 1),
        "rows": float(total_rows),
        "seconds": time.time() - start,
        "samples": samples[:10],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = ConformerT5SignPose2Text(
        text_model_name=args.text_model,
        keypoint_model_dim=args.keypoint_model_dim,
        keypoint_layers=args.keypoint_layers,
        keypoint_heads=args.keypoint_heads,
        downsample_stride=args.downsample_stride,
        dropout=args.dropout,
    ).to(device)

    dataset = SignPose2TextDataset(
        manifest=args.manifest,
        text_column=args.text_column,
        keypoint_column=args.keypoint_column,
        max_frames=args.max_frames,
        sample_mode="uniform",
        limit=args.limit,
    )
    train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    collator = SignPose2TextCollator(
        tokenizer=model.tokenizer,
        max_target_tokens=args.max_target_tokens,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    print(f"train rows : {len(train_dataset)}")
    print(f"val rows   : {len(val_dataset)}")
    print(f"device     : {device}")
    print(f"save dir   : {args.save_dir}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics = evaluate(model, val_loader, device, args)
        metrics = {"train": train_metrics, "val": val_metrics}
        print(
            f"train loss={train_metrics['loss']:.4f} "
            f"val loss={val_metrics['loss']:.4f}",
            flush=True,
        )
        for sample in val_metrics.get("samples", []):
            print(
                f"sample uid={sample['uid']} | target={sample['target']} | pred={sample['prediction']}",
                flush=True,
            )

        save_checkpoint(args.save_dir / "checkpoint_last.pt", model, optimizer, epoch, metrics, args)
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            save_checkpoint(args.save_dir / "checkpoint_best.pt", model, optimizer, epoch, metrics, args)
            print(f"new best val loss: {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
