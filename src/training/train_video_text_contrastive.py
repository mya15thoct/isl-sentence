from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.contrastive_loss import SymmetricContrastiveLoss
from src.training.video_text_dataset import (
    VideoTextEmbeddingDataset,
    collate_video_text,
    filter_usable_rows,
    read_index_csv,
    split_rows,
)
from src.video.conformer_encoder import KeypointConformerEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--text-embeddings", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument(
        "--sample-mode",
        choices=["uniform", "center", "random"],
        default="uniform",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-file-check", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    logit_scale: torch.Tensor,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "logit_scale": logit_scale.detach().cpu(),
            "metrics": metrics,
            "config": vars(args),
        },
        path,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SymmetricContrastiveLoss,
    logit_scale: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    print_every: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "v2t_acc": 0.0, "t2v_acc": 0.0, "logit_scale": 0.0}
    seen = 0
    skipped_nonfinite = 0
    start = time()

    for step, batch in enumerate(loader, start=1):
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        text_embeddings = batch["text_embeddings"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            video_embeddings = model(keypoints, lengths)
            loss, metrics = loss_fn(video_embeddings, text_embeddings, logit_scale)

        if not torch.isfinite(loss):
            skipped_nonfinite += keypoints.size(0)
            if training:
                optimizer.zero_grad(set_to_none=True)
            print(
                "warning: skipped non-finite loss "
                f"step={step} rows={seen} uids={batch['uid'][:3]}"
            )
            continue

        if training:
            assert optimizer is not None
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                logit_scale.clamp_(math.log(1.0), math.log(100.0))

        batch_size = keypoints.size(0)
        seen += batch_size
        for key in totals:
            totals[key] += metrics[key] * batch_size

        if training and print_every > 0 and step % print_every == 0:
            avg_loss = totals["loss"] / max(seen, 1)
            print(
                f"step={step} rows={seen} loss={avg_loss:.4f} "
                f"v2t={totals['v2t_acc'] / seen:.3f} "
                f"t2v={totals['t2v_acc'] / seen:.3f}"
            )

    elapsed = max(time() - start, 1e-6)
    return {
        key: value / max(seen, 1)
        for key, value in totals.items()
    } | {
        "rows": float(seen),
        "seconds": elapsed,
        "skipped_nonfinite": float(skipped_nonfinite),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    rows = read_index_csv(args.index_csv)
    rows = filter_usable_rows(rows, check_files=not args.skip_file_check)
    if not rows:
        raise SystemExit("No usable rows found. Check keypoint_path/extract_status columns.")

    text_embeddings = np.load(args.text_embeddings, mmap_mode="r")
    if text_embeddings.shape[1] != args.projection_dim:
        raise SystemExit(
            f"Text embedding dim is {text_embeddings.shape[1]}, "
            f"but projection_dim is {args.projection_dim}."
        )

    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
    train_sample_mode = args.sample_mode
    val_sample_mode = "center" if args.sample_mode == "random" else args.sample_mode

    train_dataset = VideoTextEmbeddingDataset(
        train_rows,
        text_embeddings,
        max_frames=args.max_frames,
        sample_mode=train_sample_mode,
    )
    val_dataset = VideoTextEmbeddingDataset(
        val_rows,
        text_embeddings,
        max_frames=args.max_frames,
        sample_mode=val_sample_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_video_text,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_video_text,
    )

    device = torch.device(args.device)
    model = KeypointConformerEncoder(
        model_dim=args.model_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        downsample_stride=args.downsample_stride,
        dropout=args.dropout,
    ).to(device)

    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [logit_scale],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    loss_fn = SymmetricContrastiveLoss()

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
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            logit_scale,
            device,
            optimizer,
            scaler,
            args.print_every,
        )
        print(
            "train "
            f"loss={train_metrics['loss']:.4f} "
            f"v2t={train_metrics['v2t_acc']:.3f} "
            f"t2v={train_metrics['t2v_acc']:.3f}"
        )

        val_metrics = {"loss": train_metrics["loss"], "v2t_acc": 0.0, "t2v_acc": 0.0}
        if len(val_dataset) > 0:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    loss_fn,
                    logit_scale,
                    device,
                    optimizer=None,
                    scaler=None,
                    print_every=0,
                )
            print(
                "val   "
                f"loss={val_metrics['loss']:.4f} "
                f"v2t={val_metrics['v2t_acc']:.3f} "
                f"t2v={val_metrics['t2v_acc']:.3f}"
            )

        metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(
            args.save_dir / "checkpoint_last.pt",
            model,
            optimizer,
            logit_scale,
            epoch,
            metrics,
            args,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                args.save_dir / "checkpoint_best.pt",
                model,
                optimizer,
                logit_scale,
                epoch,
                metrics,
                args,
            )


if __name__ == "__main__":
    main()
