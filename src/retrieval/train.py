"""End-to-end contrastive training for ISL pose-text retrieval.

Both encoders are fine-tuned jointly (CLIP / SignCLIP style):

  pose keypoints --> Keypoint Conformer --\
                                            >-- shared 384-d space -- InfoNCE
  caption text   --> MiniLM (trainable) --/

Extras:
  * In-batch soft positives from on-the-fly text cosine similarity, so that
    near-duplicate captions in a batch are not treated as hard negatives.
  * Optional SignCL density loss on the pose encoder's per-frame memory.

Validation reports full-set retrieval (R@1/5/10, median rank) and the best
checkpoint is selected by mean R@10.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.retrieval import PoseTextRetrievalModel
from src.retrieval.dataset import RetrievalDataset, collate_retrieval
from src.retrieval.losses import density_loss, info_nce_xbm, retrieval_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--sample-mode", choices=("uniform", "random", "center"), default="random")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-prob", type=float, default=0.8)
    parser.add_argument(
        "--augment-methods",
        nargs="+",
        default=["noise", "subsample", "scale", "crop", "jitter"],
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="pose encoder LR")
    parser.add_argument("--text-lr", type=float, default=1e-5, help="text encoder LR (pretrained)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="linear warmup before cosine decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="initial temperature for the learnable logit scale")
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.85,
        help="in-batch text cosine above this becomes a soft positive; >=1 disables",
    )
    parser.add_argument("--density-weight", type=float, default=0.0)
    parser.add_argument("--density-temperature", type=float, default=0.1)
    parser.add_argument("--density-positive-window", type=int, default=1)
    parser.add_argument("--density-negative-margin", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--pose-model-dim", type=int, default=256)
    parser.add_argument("--pose-layers", type=int, default=4)
    parser.add_argument("--pose-heads", type=int, default=4)
    parser.add_argument("--downsample-stride", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hand-aware", action="store_true", help="hand-centric encoder: hands main path + residual cross-attention from pose/face")
    parser.add_argument("--context-parts", nargs="*", choices=["pose", "face"], default=["pose", "face"], help="hand-aware ablation: which parts feed the cross-attention context (empty = hands only)")
    parser.add_argument("--motion-features", action="store_true", help="append per-frame velocity (Δ) and acceleration (ΔΔ) to positions (input becomes 3×1662); hand-aware only")
    parser.add_argument("--face-keep", action="store_true", help="zero all face landmarks except lips/eyebrows/eyes (cut face noise); input stays 1662-d")
    parser.add_argument("--no-redundancy", action="store_true", help="ablation: disable redundancy grouping (identical captions become hard negatives)")
    parser.add_argument("--queue-size", type=int, default=0, help="cross-batch memory bank size (extra contrastive negatives); 0 = off")
    parser.add_argument("--queue-warmup-epochs", type=int, default=2, help="train in-batch only for this many epochs before activating the memory bank (avoids stale-negative collapse); queue is still filled during warm-up")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay on weights; 0 = off. ~0.999 averages the last few epochs into checkpoint_ema.pt")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="warm-start from a Stage-A (word) checkpoint, loaded strict=False")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class EMA:
    """Exponential moving average of model weights (kept in fp32).

    Cheap, per-step weight averaging. Because the best epoch tends to sit at the
    very end of the cosine schedule, averaging the final epochs into one set of
    weights gives a smoother, usually slightly better model for free.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.state_dict().items()
            if param.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value.to(state[name].dtype))


class CrossBatchQueue:
    """FIFO memory bank of detached video/text embeddings for extra negatives."""

    def __init__(self, size: int, dim: int, device: torch.device) -> None:
        self.size = size
        self.video = torch.zeros(size, dim, device=device)
        self.text = torch.zeros(size, dim, device=device)
        self.groups = torch.full((size,), -1, dtype=torch.long, device=device)
        self.filled = 0
        self.ptr = 0

    @torch.no_grad()
    def get(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.filled == 0:
            return None, None, None
        return self.video[: self.filled], self.text[: self.filled], self.groups[: self.filled]

    @torch.no_grad()
    def enqueue(self, video: torch.Tensor, text: torch.Tensor, groups: torch.Tensor) -> None:
        n = video.size(0)
        idx = (torch.arange(n, device=self.video.device) + self.ptr) % self.size
        self.video[idx] = video.detach().float()
        self.text[idx] = text.detach().float()
        self.groups[idx] = groups.to(self.groups.device)
        self.ptr = int((self.ptr + n) % self.size)
        self.filled = min(self.size, self.filled + n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(dataset: RetrievalDataset, args: argparse.Namespace, shuffle: bool, device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_retrieval,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
    )


def soft_positive_mask(text_embeddings: torch.Tensor, threshold: float) -> torch.Tensor | None:
    """Off-diagonal in-batch positives where caption cosine exceeds threshold."""
    if threshold >= 1.0:
        return None
    with torch.no_grad():
        normed = F.normalize(text_embeddings.detach().float(), dim=-1, eps=1e-6)
        sim = normed @ normed.T
        mask = (sim >= threshold).float()
        mask.fill_diagonal_(0.0)
    return mask


def group_positive_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """Off-diagonal in-batch positives that share the same caption group (#1)."""
    g = group_ids.view(-1, 1)
    mask = (g == g.t()).float()
    mask.fill_diagonal_(0.0)
    return mask


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    present = [m for m in masks if m is not None]
    if not present:
        return None
    out = present[0]
    for other in present[1:]:
        out = torch.maximum(out, other)
    return out


def tokenize_batch(model: PoseTextRetrievalModel, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    tokens = model.text_encoder.tokenize(texts)
    return {key: value.to(device, non_blocking=True) for key, value in tokens.items()}


def train_one_epoch(
    model: PoseTextRetrievalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    ema: "EMA | None" = None,
    queue: "CrossBatchQueue | None" = None,
    queue_active: bool = True,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "contrastive": 0.0, "density": 0.0}
    rows = 0
    start = time.time()
    use_amp = args.amp and device.type == "cuda"

    for step, batch in enumerate(loader, start=1):
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        group_ids = batch["group_ids"].to(device, non_blocking=True)
        tokens = tokenize_batch(model, batch["texts"], device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(keypoints, lengths, tokens["input_ids"], tokens["attention_mask"])
            group_mask = None if args.no_redundancy else group_positive_mask(group_ids)
            mask = combine_masks(
                group_mask,
                soft_positive_mask(outputs["text_embedding"], args.semantic_threshold),
            )
            if queue is not None and queue_active:
                qv, qt, qg = queue.get()
                contrastive = info_nce_xbm(
                    outputs["video_embedding"],
                    outputs["text_embedding"],
                    group_ids,
                    queue_video=qv,
                    queue_text=qt,
                    queue_groups=qg,
                    in_batch_positive=mask,
                    logit_scale=model.current_logit_scale(),
                )
                loss = contrastive
                parts = {"contrastive": float(contrastive.detach().cpu()), "density": 0.0}
                if args.density_weight > 0.0:
                    den = density_loss(
                        outputs["memory"],
                        outputs["valid_mask"],
                        temperature=args.density_temperature,
                        positive_window=args.density_positive_window,
                        negative_margin=args.density_negative_margin,
                    )
                    loss = loss + args.density_weight * den
                    parts["density"] = float(den.detach().cpu())
                parts["total"] = float(loss.detach().cpu())
            else:
                loss, parts = retrieval_loss(
                    outputs["video_embedding"],
                    outputs["text_embedding"],
                    logit_scale=model.current_logit_scale(),
                    positive_mask=mask,
                    memory=outputs["memory"],
                    valid_mask=outputs["valid_mask"],
                    density_weight=args.density_weight,
                    density_temperature=args.density_temperature,
                    positive_window=args.density_positive_window,
                    negative_margin=args.density_negative_margin,
                )

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema is not None:
            ema.update(model)
        if queue is not None:
            with torch.no_grad():
                qv_new = F.normalize(outputs["video_embedding"].detach().float(), dim=-1, eps=1e-6)
                qt_new = F.normalize(outputs["text_embedding"].detach().float(), dim=-1, eps=1e-6)
            queue.enqueue(qv_new, qt_new, group_ids)

        batch_rows = int(keypoints.size(0))
        rows += batch_rows
        for key in totals:
            totals[key] += parts.get(key, 0.0) * batch_rows

        if args.print_every and step % args.print_every == 0:
            print(
                f"step={step} rows={rows} "
                f"loss={totals['total'] / rows:.4f} "
                f"con={totals['contrastive'] / rows:.4f} "
                f"den={totals['density'] / rows:.4f}",
                flush=True,
            )

    return {
        "loss": totals["total"] / max(rows, 1),
        "contrastive": totals["contrastive"] / max(rows, 1),
        "density": totals["density"] / max(rows, 1),
        "rows": float(rows),
        "seconds": time.time() - start,
    }


@torch.no_grad()
def encode_val(model: PoseTextRetrievalModel, loader: DataLoader, device: torch.device):
    model.eval()
    video_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    group_chunks: list[torch.Tensor] = []
    for batch in loader:
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        tokens = tokenize_batch(model, batch["texts"], device)
        video, _, _ = model.encode_pose(keypoints, lengths)
        text = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
        video_chunks.append(video.float().cpu())
        text_chunks.append(text.float().cpu())
        group_chunks.append(batch["group_ids"])
    video = F.normalize(torch.cat(video_chunks, dim=0), dim=-1, eps=1e-6)
    text = F.normalize(torch.cat(text_chunks, dim=0), dim=-1, eps=1e-6)
    groups = torch.cat(group_chunks, dim=0)
    return video, text, groups


def rank_metrics(query: torch.Tensor, candidates: torch.Tensor, chunk_size: int) -> dict[str, float]:
    """Exact retrieval: only the paired (diagonal) caption counts as correct."""
    n = query.size(0)
    ranks: list[int] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores = query[start:end] @ candidates.T
        target = scores[:, start:end].diag().unsqueeze(1)
        rank = (scores > target).sum(dim=1) + 1
        ranks.extend(rank.tolist())
    return _rank_summary(ranks)


def redundancy_rank_metrics(
    query: torch.Tensor,
    candidates: torch.Tensor,
    group_ids: torch.Tensor,
    chunk_size: int,
) -> dict[str, float]:
    """Redundancy-aware retrieval (#1): any caption in the same group counts as
    correct, so identical captions are not penalized as misses."""
    n = query.size(0)
    ranks: list[int] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores = query[start:end] @ candidates.T
        relevant = group_ids[start:end].view(-1, 1) == group_ids.view(1, -1)
        neg_inf = torch.finfo(scores.dtype).min
        best_relevant = scores.masked_fill(~relevant, neg_inf).max(dim=1).values
        rank = (scores > best_relevant.unsqueeze(1)).sum(dim=1) + 1
        ranks.extend(rank.tolist())
    return _rank_summary(ranks)


def _rank_summary(ranks: list[int]) -> dict[str, float]:
    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    return {
        "r1": float((ranks_t <= 1).float().mean()),
        "r5": float((ranks_t <= 5).float().mean()),
        "r10": float((ranks_t <= 10).float().mean()),
        "median_rank": float(ranks_t.median()),
    }


def evaluate(model: PoseTextRetrievalModel, loader: DataLoader, device: torch.device, chunk: int) -> dict[str, Any]:
    video, text, groups = encode_val(model, loader, device)
    v2t = rank_metrics(video, text, chunk)
    t2v = rank_metrics(text, video, chunk)
    rv2t = redundancy_rank_metrics(video, text, groups, chunk)
    rt2v = redundancy_rank_metrics(text, video, groups, chunk)
    return {
        "rows": int(video.size(0)),
        "groups": int(groups.unique().numel()),
        **{f"v2t_{k}": v for k, v in v2t.items()},
        **{f"t2v_{k}": v for k, v in t2v.items()},
        **{f"rv2t_{k}": v for k, v in rv2t.items()},
        **{f"rt2v_{k}": v for k, v in rt2v.items()},
        "mean_r10": 0.5 * (v2t["r10"] + t2v["r10"]),
        "rmean_r10": 0.5 * (rv2t["r10"] + rt2v["r10"]),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args: argparse.Namespace) -> None:
    # Optimizer state is intentionally NOT saved: this project never resumes
    # training, only loads model_state for eval/ensemble, so storing it just
    # tripled checkpoint size (~3.2GB -> ~1.4GB) and filled the disk.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a failed write can't corrupt the previous checkpoint


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = PoseTextRetrievalModel(
        embedding_dim=args.embedding_dim,
        pose_model_dim=args.pose_model_dim,
        pose_layers=args.pose_layers,
        pose_heads=args.pose_heads,
        downsample_stride=args.downsample_stride,
        dropout=args.dropout,
        hand_aware=args.hand_aware,
        context_parts=tuple(args.context_parts),
        motion=args.motion_features,
        text_model_name=args.text_model,
        max_text_length=args.max_text_length,
    ).to(device)

    if args.init_checkpoint is not None:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model_state = state.get("model_state", state) if isinstance(state, dict) else state
        result = model.load_state_dict(model_state, strict=False)
        print(
            f"warm-start from {args.init_checkpoint}: "
            f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
            flush=True,
        )

    train_dataset = RetrievalDataset(
        manifest=args.manifest,
        text_column=args.text_column,
        keypoint_column=args.keypoint_column,
        max_frames=args.max_frames,
        sample_mode=args.sample_mode,
        limit=args.limit,
        augment=args.augment,
        augment_probability=args.augment_prob,
        augment_methods=args.augment_methods,
        motion_features=args.motion_features,
        face_keep=args.face_keep,
    )
    val_dataset = RetrievalDataset(
        manifest=args.val_manifest,
        text_column=args.text_column,
        keypoint_column=args.keypoint_column,
        max_frames=args.max_frames,
        sample_mode="uniform",
        limit=args.limit,
        motion_features=args.motion_features,
        face_keep=args.face_keep,
    )
    train_loader = build_loader(train_dataset, args, shuffle=True, device=device)
    val_loader = build_loader(val_dataset, args, shuffle=False, device=device)

    text_params = list(model.text_encoder.parameters())
    text_ids = {id(p) for p in text_params}
    other_params = [p for p in model.parameters() if id(p) not in text_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": text_params, "lr": args.text_lr},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    ema = EMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    queue = (
        CrossBatchQueue(args.queue_size, args.embedding_dim, device)
        if args.queue_size > 0
        else None
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(f"train rows : {len(train_dataset)}")
    print(f"val rows   : {len(val_dataset)}")
    print(f"device     : {device}  amp={args.amp}")
    print(f"pose LR={args.lr}  text LR={args.text_lr}  density_w={args.density_weight}")
    print(f"queue_size={args.queue_size}  ema_decay={args.ema_decay}")

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        queue_active = queue is not None and epoch > args.queue_warmup_epochs
        if queue is not None:
            print(f"  memory bank: {'ON' if queue_active else f'warm-up (in-batch only, until epoch {args.queue_warmup_epochs})'}", flush=True)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, args,
            ema=ema, queue=queue, queue_active=queue_active,
        )
        val_metrics = evaluate(model, val_loader, device, args.eval_chunk_size)
        metrics = {"train": train_metrics, "val": val_metrics}
        print(
            f"train loss={train_metrics['loss']:.4f} con={train_metrics['contrastive']:.4f} "
            f"den={train_metrics['density']:.4f} lr={scheduler.get_last_lr()[0]:.2e} | "
            f"val v2t r1/5/10={val_metrics['v2t_r1']:.3f}/{val_metrics['v2t_r5']:.3f}/{val_metrics['v2t_r10']:.3f} "
            f"t2v r10={val_metrics['t2v_r10']:.3f} med={val_metrics['v2t_median_rank']:.0f} "
            f"mean_r10={val_metrics['mean_r10']:.3f} rmean_r10={val_metrics['rmean_r10']:.3f}",
            flush=True,
        )
        save_checkpoint(args.save_dir / "checkpoint_last.pt", model, optimizer, epoch, metrics, args)
        if val_metrics["mean_r10"] > best:
            best = val_metrics["mean_r10"]
            save_checkpoint(args.save_dir / "checkpoint_best.pt", model, optimizer, epoch, metrics, args)
            print(f"new best mean_r10: {best:.4f}", flush=True)

    if ema is not None:
        ema.copy_to(model)
        ema_metrics = evaluate(model, val_loader, device, args.eval_chunk_size)
        print(
            f"EMA val v2t r1/5/10={ema_metrics['v2t_r1']:.3f}/{ema_metrics['v2t_r5']:.3f}/{ema_metrics['v2t_r10']:.3f} "
            f"mean_r10={ema_metrics['mean_r10']:.3f} rmean_r10={ema_metrics['rmean_r10']:.3f}",
            flush=True,
        )
        save_checkpoint(
            args.save_dir / "checkpoint_ema.pt", model, optimizer, args.epochs,
            {"val": ema_metrics}, args,
        )
        print("saved EMA weights to checkpoint_ema.pt", flush=True)


if __name__ == "__main__":
    main()
