from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
from time import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.keypoint_augmentation import augment_sequence, augmentation_methods
from src.training.video_text_dataset import KEYPOINT_DIM, sample_frames
from src.video.conformer_encoder import KeypointConformerEncoder


DEFAULTS = {
    "model_dim": 256,
    "projection_dim": 384,
    "num_layers": 4,
    "num_heads": 4,
    "downsample_stride": 4,
    "dropout": 0.1,
}
PRETRAINED_ARCH_FIELDS = [
    "model_dim",
    "projection_dim",
    "num_layers",
    "num_heads",
    "downsample_stride",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune/evaluate a sentence classifier on the 491-video target set.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument(
        "--sample-mode",
        choices=["uniform", "center", "random"],
        default="uniform",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum val top1 improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--lr-reduce-patience",
        type=int,
        default=0,
        help="If >0, reduce learning rates after this many non-improving epochs.",
    )
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--classifier", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--logit-scale-init", type=float, default=10.0)
    parser.add_argument("--max-logit-scale", type=float, default=100.0)
    parser.add_argument(
        "--class-text-embeddings",
        type=Path,
        help="Optional (num_classes, projection_dim) text/gloss embedding matrix for logit blending.",
    )
    parser.add_argument(
        "--text-logit-weight",
        type=float,
        default=0.0,
        help="Add this weight times class-text logits to classifier logits.",
    )
    parser.add_argument(
        "--text-loss-weight",
        type=float,
        default=0.0,
        help="Optional auxiliary CE loss on text-prototype logits alone.",
    )
    parser.add_argument("--text-logit-scale", type=float, default=20.0)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="Freeze the pretrained encoder for this many initial epochs, then unfreeze.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--augment-methods",
        nargs="+",
        choices=augmentation_methods(),
        default=["noise", "subsample", "scale", "crop", "jitter"],
        help="Train-only keypoint augmentation methods. Temporal reverse is intentionally omitted.",
    )
    parser.add_argument("--augment-prob", type=float, default=0.75)
    parser.add_argument("--train-repeat", type=int, default=1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--jitter-std", type=float, default=0.02)
    parser.add_argument("--subsample-min", type=float, default=0.75)
    parser.add_argument("--subsample-max", type=float, default=0.95)
    parser.add_argument("--temporal-scale-min", type=float, default=0.85)
    parser.add_argument("--temporal-scale-max", type=float, default=1.20)
    parser.add_argument("--crop-min", type=float, default=0.85)
    parser.add_argument("--crop-max", type=float, default=0.98)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--downsample-stride", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"keypoint_path", "label_id"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def build_label_names(rows: list[dict[str, str]]) -> list[str]:
    label_to_text: dict[int, str] = {}
    for row in rows:
        label_id = int(row["label_id"])
        label_to_text.setdefault(
            label_id,
            row.get("label_text") or row.get("sentence") or str(label_id),
        )
    return [label_to_text[idx] for idx in sorted(label_to_text)]


def stratified_split(
    rows: list[dict[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(int(row["label_id"]), []).append(row)

    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for class_rows in by_class.values():
        class_rows = list(class_rows)
        rng.shuffle(class_rows)
        n = len(class_rows)

        test_n = 1 if test_ratio > 0 and n >= 3 else 0
        val_n = 1 if val_ratio > 0 and n - test_n >= 3 else 0

        # For larger classes, allow a slightly larger split while preserving train rows.
        if n >= 8:
            test_n = max(test_n, int(round(n * test_ratio)))
            val_n = max(val_n, int(round(n * val_ratio)))
            while test_n + val_n > n - 2:
                if test_n >= val_n and test_n > 1:
                    test_n -= 1
                elif val_n > 1:
                    val_n -= 1
                else:
                    break

        test.extend(class_rows[:test_n])
        val.extend(class_rows[test_n : test_n + val_n])
        train.extend(class_rows[test_n + val_n :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class TargetSentenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        max_frames: int,
        sample_mode: str,
        augment: bool = False,
        augment_methods: list[str] | None = None,
        augment_prob: float = 0.75,
        repeat: int = 1,
        noise_std: float = 0.01,
        jitter_std: float = 0.02,
        subsample_min: float = 0.75,
        subsample_max: float = 0.95,
        temporal_scale_min: float = 0.85,
        temporal_scale_max: float = 1.20,
        crop_min: float = 0.85,
        crop_max: float = 0.98,
    ) -> None:
        self.rows = rows
        self.max_frames = max_frames
        self.sample_mode = sample_mode
        self.augment = augment
        self.augment_methods = augment_methods or []
        self.augment_prob = augment_prob
        self.repeat = max(1, repeat)
        self.noise_std = noise_std
        self.jitter_std = jitter_std
        self.subsample_min = subsample_min
        self.subsample_max = subsample_max
        self.temporal_scale_min = temporal_scale_min
        self.temporal_scale_max = temporal_scale_max
        self.crop_min = crop_min
        self.crop_max = crop_max

    def __len__(self) -> int:
        return len(self.rows) * self.repeat

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index % len(self.rows)]
        keypoints = np.load(row["keypoint_path"]).astype(np.float32, copy=False)
        if keypoints.ndim != 2 or keypoints.shape[1] != KEYPOINT_DIM:
            raise ValueError(f"Invalid keypoint shape {keypoints.shape}: {row['keypoint_path']}")
        keypoints = sample_frames(keypoints, self.max_frames, self.sample_mode).astype(
            np.float32,
            copy=True,
        )
        np.nan_to_num(keypoints, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.augment:
            keypoints = augment_sequence(
                keypoints,
                methods=self.augment_methods,
                probability=self.augment_prob,
                noise_std=self.noise_std,
                jitter_std=self.jitter_std,
                subsample_min=self.subsample_min,
                subsample_max=self.subsample_max,
                scale_min=self.temporal_scale_min,
                scale_max=self.temporal_scale_max,
                crop_min=self.crop_min,
                crop_max=self.crop_max,
            )
        return {
            "uid": row.get("uid", ""),
            "label": int(row["label_id"]),
            "keypoints": torch.from_numpy(keypoints),
            "length": keypoints.shape[0],
        }


def collate_target(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feature_dim = batch[0]["keypoints"].shape[1]
    keypoints = torch.zeros(len(batch), max_len, feature_dim, dtype=torch.float32)
    for i, item in enumerate(batch):
        x = item["keypoints"]
        keypoints[i, : x.shape[0]] = x
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    return {
        "uid": [item["uid"] for item in batch],
        "keypoints": keypoints,
        "lengths": lengths,
        "labels": labels,
    }


class SentenceClassifier(nn.Module):
    def __init__(
        self,
        encoder: KeypointConformerEncoder,
        embedding_dim: int,
        num_classes: int,
        classifier: str = "linear",
        head_dropout: float = 0.0,
        logit_scale_init: float = 10.0,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        if classifier == "linear":
            self.classifier = nn.Linear(embedding_dim, num_classes)
        elif classifier == "cosine":
            self.classifier = CosineClassifier(
                embedding_dim,
                num_classes,
                logit_scale_init=logit_scale_init,
                max_logit_scale=max_logit_scale,
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

    def forward(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(keypoints, lengths)
        logits = self.classifier(self.head_dropout(embedding))
        if return_embedding:
            return logits, embedding
        return logits


class CosineClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        logit_scale_init: float = 10.0,
        max_logit_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init), dtype=torch.float32))
        self.max_logit_scale = max_logit_scale

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = F.normalize(embedding.float(), dim=-1, eps=1e-6)
        weight = F.normalize(self.weight.float(), dim=-1, eps=1e-6)
        scale = self.logit_scale.float().exp().clamp(max=self.max_logit_scale)
        return scale * embedding @ weight.t()


def cfg_value(config: dict[str, Any], name: str, fallback: Any) -> Any:
    return config.get(name, fallback)


def read_pretrained_config(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("config", {}) or {}


def load_pretrained_encoder(
    encoder: KeypointConformerEncoder,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = encoder.load_state_dict(checkpoint["model_state"], strict=False)
    bad = [
        name
        for name, tensor in checkpoint["model_state"].items()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all()
    ]
    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
        "bad_tensors": bad,
        "config": checkpoint.get("config", {}) or {},
    }


def set_encoder_trainable(model: SentenceClassifier, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def build_optimizer(
    model: SentenceClassifier,
    lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    param_groups = []
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr})
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    param_groups.append({"params": head_params, "lr": head_lr})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def format_lrs(optimizer: torch.optim.Optimizer) -> str:
    return ", ".join(f"{group['lr']:.3g}" for group in optimizer.param_groups)


def reduce_learning_rates(
    optimizer: torch.optim.Optimizer,
    factor: float,
    min_lr: float,
) -> list[tuple[float, float]]:
    changes: list[tuple[float, float]] = []
    for group in optimizer.param_groups:
        old_lr = float(group["lr"])
        new_lr = max(old_lr * factor, min_lr)
        if new_lr < old_lr:
            group["lr"] = new_lr
            changes.append((old_lr, new_lr))
    return changes


def load_class_text_embeddings(
    path: Path | None,
    num_classes: int,
    projection_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if path is None:
        return None
    embeddings = np.load(path).astype(np.float32, copy=True)
    if embeddings.shape != (num_classes, projection_dim):
        raise SystemExit(
            "Class text embedding shape mismatch: "
            f"expected {(num_classes, projection_dim)}, got {embeddings.shape}"
        )
    np.nan_to_num(embeddings, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    return F.normalize(tensor, dim=-1, eps=1e-6)


def make_loaders(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TargetSentenceDataset(
        train_rows,
        args.max_frames,
        args.sample_mode,
        augment=args.augment,
        augment_methods=args.augment_methods,
        augment_prob=args.augment_prob,
        repeat=args.train_repeat,
        noise_std=args.noise_std,
        jitter_std=args.jitter_std,
        subsample_min=args.subsample_min,
        subsample_max=args.subsample_max,
        temporal_scale_min=args.temporal_scale_min,
        temporal_scale_max=args.temporal_scale_max,
        crop_min=args.crop_min,
        crop_max=args.crop_max,
    )
    eval_mode = "center" if args.sample_mode == "random" else args.sample_mode
    val_dataset = TargetSentenceDataset(val_rows, args.max_frames, eval_mode)
    test_dataset = TargetSentenceDataset(test_rows, args.max_frames, eval_mode)

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_target,
    }
    return (
        DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args),
        DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args),
        DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args),
    )


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.size(1))
    pred = logits.topk(k, dim=1).indices
    return float((pred == labels[:, None]).any(dim=1).float().mean().item())


def run_epoch(
    model: SentenceClassifier,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    print_every: int,
    label_smoothing: float = 0.0,
    class_text_embeddings: torch.Tensor | None = None,
    text_logit_weight: float = 0.0,
    text_loss_weight: float = 0.0,
    text_logit_scale: float = 20.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    seen = 0
    start = time()

    for step, batch in enumerate(loader, start=1):
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            model_output = model(
                keypoints,
                lengths,
                return_embedding=class_text_embeddings is not None,
            )
            if class_text_embeddings is None:
                logits = model_output
                text_logits = None
            else:
                logits, embeddings = model_output
                text_logits = text_logit_scale * (
                    F.normalize(embeddings.float(), dim=-1, eps=1e-6)
                    @ class_text_embeddings.t()
                )
                if text_logit_weight > 0:
                    logits = logits + text_logit_weight * text_logits
            loss = F.cross_entropy(
                logits.float(),
                labels,
                label_smoothing=label_smoothing,
            )
            if text_logits is not None and text_loss_weight > 0:
                loss = loss + text_loss_weight * F.cross_entropy(
                    text_logits.float(),
                    labels,
                    label_smoothing=label_smoothing,
                )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite classifier loss at step={step} uids={batch['uid'][:5]}")

        if training:
            assert optimizer is not None
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        seen += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_top1 += topk_accuracy(logits.detach(), labels, 1) * batch_size
        total_top5 += topk_accuracy(logits.detach(), labels, 5) * batch_size

        if training and print_every > 0 and step % print_every == 0:
            print(
                f"step={step} rows={seen} "
                f"loss={total_loss / seen:.4f} "
                f"top1={total_top1 / seen:.3f} "
                f"top5={total_top5 / seen:.3f}"
            )

    elapsed = max(time() - start, 1e-6)
    return {
        "loss": total_loss / max(seen, 1),
        "top1": total_top1 / max(seen, 1),
        "top5": total_top5 / max(seen, 1),
        "rows": float(seen),
        "seconds": elapsed,
    }


def save_checkpoint(
    path: Path,
    model: SentenceClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    label_names: list[str],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "encoder_state": model.encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "label_names": label_names,
            "config": vars(args),
        },
        path,
    )


def write_split(path: Path, split_name: str, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{split_name} split saved: {path}")


def main() -> None:
    args = parse_args()
    if args.lr_reduce_patience > 0:
        if not 0.0 < args.lr_reduce_factor < 1.0:
            raise SystemExit("--lr-reduce-factor must be between 0 and 1 when LR reduction is enabled.")
        if args.min_lr <= 0:
            raise SystemExit("--min-lr must be positive.")
    set_seed(args.seed)
    rows = read_manifest(args.manifest)
    label_names = build_label_names(rows)
    num_classes = len(label_names)
    if num_classes < 2:
        raise SystemExit("Need at least two classes.")

    train_rows, val_rows, test_rows = stratified_split(
        rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    if not train_rows or not val_rows or not test_rows:
        raise SystemExit(
            f"Bad split sizes: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )

    if args.pretrained_checkpoint is not None:
        pretrained_config = read_pretrained_config(args.pretrained_checkpoint)
        for name in PRETRAINED_ARCH_FIELDS:
            if name in pretrained_config:
                setattr(args, name.replace("-", "_"), cfg_value(pretrained_config, name, getattr(args, name)))

    device = torch.device(args.device)
    encoder = KeypointConformerEncoder(
        model_dim=args.model_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        downsample_stride=args.downsample_stride,
        dropout=args.dropout,
    ).to(device)

    pretrained_info: dict[str, Any] = {}
    if args.pretrained_checkpoint is not None:
        pretrained_info = load_pretrained_encoder(encoder, args.pretrained_checkpoint, device)
        print(f"loaded pretrained: {args.pretrained_checkpoint}")
        print(f"pretrained epoch : {pretrained_info.get('epoch')}")
        print(f"bad tensors      : {len(pretrained_info.get('bad_tensors', []))}")
        if pretrained_info.get("missing_keys") or pretrained_info.get("unexpected_keys"):
            print(f"missing keys     : {pretrained_info.get('missing_keys', [])[:10]}")
            print(f"unexpected keys  : {pretrained_info.get('unexpected_keys', [])[:10]}")

    initially_freeze_encoder = args.freeze_encoder or args.freeze_epochs > 0
    if initially_freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False

    model = SentenceClassifier(
        encoder,
        args.projection_dim,
        num_classes,
        classifier=args.classifier,
        head_dropout=args.head_dropout,
        logit_scale_init=args.logit_scale_init,
        max_logit_scale=args.max_logit_scale,
    ).to(device)
    class_text_embeddings = load_class_text_embeddings(
        args.class_text_embeddings,
        num_classes,
        args.projection_dim,
        device,
    )
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    optimizer = build_optimizer(model, args.lr, head_lr, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    train_loader, val_loader, test_loader = make_loaders(train_rows, val_rows, test_rows, args)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )
    (args.save_dir / "label_names.json").write_text(
        json.dumps(label_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_split(args.save_dir / "split_train.csv", "train", train_rows)
    write_split(args.save_dir / "split_val.csv", "val", val_rows)
    write_split(args.save_dir / "split_test.csv", "test", test_rows)

    print(f"classes    : {num_classes}")
    print(f"train rows : {len(train_rows)}")
    print(f"val rows   : {len(val_rows)}")
    print(f"test rows  : {len(test_rows)}")
    print(f"device     : {device}")
    print(f"freeze enc : {args.freeze_encoder}")
    print(f"freeze epochs: {args.freeze_epochs}")
    print(f"classifier : {args.classifier}")
    print(f"head dropout: {args.head_dropout}")
    print(f"lrs        : {format_lrs(optimizer)}")
    print(
        "lr reduce  : "
        f"patience={args.lr_reduce_patience} "
        f"factor={args.lr_reduce_factor} min_lr={args.min_lr}"
    )
    print(f"class text : {args.class_text_embeddings or ''}")
    print(f"text weight: {args.text_logit_weight} aux={args.text_loss_weight}")
    print(f"augment    : {args.augment} methods={args.augment_methods} prob={args.augment_prob}")
    print(f"train repeat: {args.train_repeat}")
    print(f"label smoothing: {args.label_smoothing}")
    print(f"save dir   : {args.save_dir}")

    best_val = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if (
            args.freeze_epochs > 0
            and not args.freeze_encoder
            and epoch == args.freeze_epochs + 1
        ):
            print(f"unfreezing encoder at epoch {epoch}")
            set_encoder_trainable(model, True)
            optimizer = build_optimizer(model, args.lr, head_lr, args.weight_decay)
            print(f"optimizer rebuilt with lrs: {format_lrs(optimizer)}")

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            print_every=args.print_every,
            label_smoothing=args.label_smoothing,
            class_text_embeddings=class_text_embeddings,
            text_logit_weight=args.text_logit_weight,
            text_loss_weight=args.text_loss_weight,
            text_logit_scale=args.text_logit_scale,
        )
        print(
            "train "
            f"loss={train_metrics['loss']:.4f} "
            f"top1={train_metrics['top1']:.3f} "
            f"top5={train_metrics['top5']:.3f}"
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                scaler=None,
                print_every=0,
                label_smoothing=0.0,
                class_text_embeddings=class_text_embeddings,
                text_logit_weight=args.text_logit_weight,
                text_loss_weight=0.0,
                text_logit_scale=args.text_logit_scale,
            )
        print(
            "val   "
            f"loss={val_metrics['loss']:.4f} "
            f"top1={val_metrics['top1']:.3f} "
            f"top5={val_metrics['top5']:.3f}"
        )

        metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(metrics)
        save_checkpoint(
            args.save_dir / "checkpoint_last.pt",
            model,
            optimizer,
            epoch,
            metrics,
            label_names,
            args,
        )
        improved = val_metrics["top1"] > best_val + args.early_stop_min_delta
        if improved:
            best_val = val_metrics["top1"]
            epochs_without_improvement = 0
            save_checkpoint(
                args.save_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                metrics,
                label_names,
                args,
            )
        else:
            epochs_without_improvement += 1
            print(
                f"early-stop no_improvement={epochs_without_improvement}/"
                f"{args.early_stop_patience} best_val_top1={best_val:.3f}"
            )
            if (
                args.lr_reduce_patience > 0
                and epochs_without_improvement >= args.lr_reduce_patience
            ):
                changes = reduce_learning_rates(
                    optimizer,
                    factor=args.lr_reduce_factor,
                    min_lr=args.min_lr,
                )
                if changes:
                    change_text = ", ".join(
                        f"{old_lr:.3g}->{new_lr:.3g}" for old_lr, new_lr in changes
                    )
                    print(f"lr reduced: {change_text}")
                    epochs_without_improvement = 0
                else:
                    print(f"lr already at min_lr={args.min_lr}")
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch}")
                break

    best_path = args.save_dir / "checkpoint_best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            device,
            optimizer=None,
            scaler=None,
            print_every=0,
            label_smoothing=0.0,
            class_text_embeddings=class_text_embeddings,
            text_logit_weight=args.text_logit_weight,
            text_loss_weight=0.0,
            text_logit_scale=args.text_logit_scale,
        )
    print(
        "test  "
        f"loss={test_metrics['loss']:.4f} "
        f"top1={test_metrics['top1']:.3f} "
        f"top5={test_metrics['top5']:.3f}"
    )

    summary = {
        "best_checkpoint": str(best_path),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val": best_checkpoint["metrics"]["val"],
        "test": test_metrics,
        "history": history,
        "pretrained": {
            "path": str(args.pretrained_checkpoint) if args.pretrained_checkpoint else "",
            "info": pretrained_info,
        },
    }
    (args.save_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"summary saved: {args.save_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
