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
    parser.add_argument(
        "--split-group-key",
        default="",
        help=(
            "Keep rows with the same group value in the same split. "
            "Use video_uid for relation-aware manifests to avoid train/val leakage."
        ),
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument(
        "--sample-mode",
        choices=["uniform", "center", "random"],
        default="uniform",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Initialize model weights from a checkpoint, but start a fresh optimizer/run.",
    )
    parser.add_argument("--skip-nonfinite-batches", action="store_true")
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


def sanitize_model_state(model: nn.Module) -> list[str]:
    """Repair non-finite floating point parameters/buffers after a bad batch."""
    repaired: list[str] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_floating_point() and not torch.isfinite(param).all():
                repaired.append(name)
                param.copy_(torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0))

        for name, buffer in model.named_buffers():
            if not buffer.is_floating_point() or torch.isfinite(buffer).all():
                continue

            repaired.append(name)
            if name.endswith("running_var"):
                buffer.copy_(torch.nan_to_num(buffer, nan=1.0, posinf=1.0, neginf=1.0))
                buffer.clamp_(min=1e-6)
            else:
                buffer.copy_(torch.nan_to_num(buffer, nan=0.0, posinf=0.0, neginf=0.0))

    return repaired


def format_tensor_stats(name: str, tensor: torch.Tensor) -> str:
    value = tensor.detach()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    total = finite.numel()
    if finite_count == 0:
        return f"{name}: shape={tuple(value.shape)} finite=0/{total}"

    finite_values = value[finite].float()
    return (
        f"{name}: shape={tuple(value.shape)} finite={finite_count}/{total} "
        f"min={finite_values.min().item():.6g} "
        f"max={finite_values.max().item():.6g} "
        f"mean={finite_values.mean().item():.6g}"
    )


def group_values(rows: list[dict[str, str]], group_key: str) -> set[str]:
    values: set[str] = set()
    for idx, row in enumerate(rows):
        value = (
            row.get(group_key, "").strip()
            or row.get("keypoint_path", "").strip()
            or row.get("uid", "").strip()
            or f"row-{idx}"
        )
        values.add(value)
    return values


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SymmetricContrastiveLoss,
    logit_scale: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    print_every: int,
    skip_nonfinite_batches: bool,
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
        relation_weights = batch["relation_weights"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            video_embeddings = model(keypoints, lengths)
            loss, metrics = loss_fn(
                video_embeddings,
                text_embeddings,
                logit_scale,
                sample_weights=relation_weights,
            )

        if not torch.isfinite(loss):
            if training:
                optimizer.zero_grad(set_to_none=True)
            details = "\n".join(
                [
                    f"non-finite loss at step={step} rows={seen}",
                    f"uids={batch['uid']}",
                    f"texts={batch['text']}",
                    f"relations={batch.get('relation_type', [])}",
                    format_tensor_stats("keypoints", keypoints),
                    format_tensor_stats("lengths", lengths),
                    format_tensor_stats("relation_weights", relation_weights),
                    format_tensor_stats("video_embeddings", video_embeddings),
                    format_tensor_stats("text_embeddings", text_embeddings),
                    format_tensor_stats("logit_scale", logit_scale),
                ]
            )
            if not skip_nonfinite_batches:
                raise RuntimeError(details)

            skipped_nonfinite += keypoints.size(0)
            repaired = sanitize_model_state(model)
            print(
                "warning: skipped non-finite loss "
                f"step={step} rows={seen} uids={batch['uid'][:3]} "
                f"repaired={repaired[:5]}"
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
                logit_scale.copy_(
                    torch.nan_to_num(
                        logit_scale,
                        nan=math.log(1 / 0.07),
                        posinf=math.log(100.0),
                        neginf=math.log(1.0),
                    )
                )
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

    train_rows, val_rows = split_rows(
        rows,
        args.val_ratio,
        args.seed,
        group_key=args.split_group_key or None,
    )
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
    start_epoch = 1
    best_val = float("inf")
    epochs_without_improvement = 0

    if args.resume_from is not None and args.init_from is not None:
        raise SystemExit("Use either --resume-from or --init-from, not both.")

    if args.init_from is not None:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
        if "logit_scale" in checkpoint:
            logit_scale.data.copy_(checkpoint["logit_scale"].to(device=device, dtype=torch.float32))
        repaired = sanitize_model_state(model)
        print(f"initialized from: {args.init_from}")
        print(f"init epoch      : {checkpoint.get('epoch')}")
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "warning: checkpoint key mismatch "
                f"missing={load_result.missing_keys[:10]} "
                f"unexpected={load_result.unexpected_keys[:10]}"
            )
        if repaired:
            print(f"warning: repaired checkpoint tensors: {repaired[:10]}")

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        logit_scale.data.copy_(checkpoint["logit_scale"].to(device=device, dtype=torch.float32))
        repaired = sanitize_model_state(model)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        val_loss = checkpoint.get("metrics", {}).get("val", {}).get("loss", float("inf"))
        if math.isfinite(float(val_loss)):
            best_val = float(val_loss)
        print(f"resumed from: {args.resume_from}")
        print(f"resume epoch: {start_epoch}")
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "warning: checkpoint key mismatch "
                f"missing={load_result.missing_keys[:10]} "
                f"unexpected={load_result.unexpected_keys[:10]}"
            )
        if repaired:
            print(f"warning: repaired checkpoint tensors: {repaired[:10]}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    print(f"train rows : {len(train_dataset)}")
    print(f"val rows   : {len(val_dataset)}")
    print(f"device     : {device}")
    print(f"save dir   : {args.save_dir}")
    if args.split_group_key:
        print(f"split group: {args.split_group_key}")
        train_groups = group_values(train_rows, args.split_group_key)
        val_groups = group_values(val_rows, args.split_group_key)
        overlap = train_groups & val_groups
        print(f"train groups: {len(train_groups)}")
        print(f"val groups  : {len(val_groups)}")
        print(f"group overlap: {len(overlap)}")
        if overlap:
            print(f"warning: split group overlap sample={sorted(overlap)[:5]}")

    if start_epoch > args.epochs:
        print(f"checkpoint is already at epoch {start_epoch - 1}; nothing to train.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
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
            args.skip_nonfinite_batches,
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
                    skip_nonfinite_batches=args.skip_nonfinite_batches,
                )
            print(
                "val   "
                f"loss={val_metrics['loss']:.4f} "
                f"v2t={val_metrics['v2t_acc']:.3f} "
                f"t2v={val_metrics['t2v_acc']:.3f}"
            )

        metrics = {"train": train_metrics, "val": val_metrics}
        repaired = sanitize_model_state(model)
        if repaired:
            print(f"warning: repaired model tensors before checkpoint: {repaired[:10]}")
        save_checkpoint(
            args.save_dir / "checkpoint_last.pt",
            model,
            optimizer,
            logit_scale,
            epoch,
            metrics,
            args,
        )
        improved = val_metrics["loss"] < best_val - args.early_stop_min_delta
        if improved:
            best_val = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                args.save_dir / "checkpoint_best.pt",
                model,
                optimizer,
                logit_scale,
                epoch,
                metrics,
                args,
            )
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0:
                print(
                    "early-stop "
                    f"no_improvement={epochs_without_improvement}/"
                    f"{args.early_stop_patience} best_val={best_val:.4f}"
                )
                if epochs_without_improvement >= args.early_stop_patience:
                    print(f"early stopping at epoch {epoch}")
                    break


if __name__ == "__main__":
    main()
