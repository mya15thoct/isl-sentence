"""Generate SignPose2Text predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modeling.signpose2text import ConformerT5SignPose2Text
from src.training.signpose2text_dataset import (
    SignPose2TextCollator,
    SignPose2TextDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--keypoint-column", default=None)
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--print-every", type=int, default=50)
    return parser.parse_args()


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config") or {}
    return dict(config)


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val = sorted(indices[:val_size])
    train = sorted(indices[val_size:])
    return train, val


def select_split(
    dataset: SignPose2TextDataset,
    split: str,
    val_ratio: float,
    seed: int,
) -> SignPose2TextDataset | Subset:
    if split == "all":
        return dataset
    train_idx, val_idx = split_indices(len(dataset), val_ratio, seed)
    return Subset(dataset, val_idx if split == "val" else train_idx)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_config(checkpoint)

    text_model = args.text_model or config.get("text_model", "t5-small")
    text_column = args.text_column or config.get("text_column", "canonical_text")
    keypoint_column = args.keypoint_column or config.get("keypoint_column", "keypoint_path")
    max_frames = args.max_frames or int(config.get("max_frames", 512))
    max_target_tokens = args.max_target_tokens or int(config.get("max_target_tokens", 96))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(config.get("val_ratio", 0.02))
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = ConformerT5SignPose2Text(
        text_model_name=text_model,
        keypoint_model_dim=int(config.get("keypoint_model_dim", 256)),
        keypoint_layers=int(config.get("keypoint_layers", 4)),
        keypoint_heads=int(config.get("keypoint_heads", 4)),
        downsample_stride=int(config.get("downsample_stride", 4)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "warning: checkpoint key mismatch "
            f"missing={load_result.missing_keys[:8]} "
            f"unexpected={load_result.unexpected_keys[:8]}",
            flush=True,
        )
    model.eval()

    base_dataset = SignPose2TextDataset(
        manifest=args.manifest,
        text_column=text_column,
        keypoint_column=keypoint_column,
        max_frames=max_frames,
        sample_mode="uniform",
        limit=args.limit,
    )
    dataset = select_split(base_dataset, args.split, val_ratio, seed)
    collator = SignPose2TextCollator(
        tokenizer=model.tokenizer,
        max_target_tokens=max_target_tokens,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    print(f"rows selected: {len(dataset)}")
    print(f"split        : {args.split}")
    print(f"num beams    : {args.num_beams}")
    print(f"output       : {args.output}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["uid", "source_row", "target", "prediction"]
    rows_predicted = 0
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        output_file.flush()

        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                keypoints = batch["keypoints"].to(device, non_blocking=True)
                lengths = batch["lengths"].to(device, non_blocking=True)
                generated = model.generate(
                    keypoints=keypoints,
                    lengths=lengths,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
                predictions = model.decode(generated)
                for uid, source_row, target, prediction in zip(
                    batch["uids"],
                    batch["source_rows"].tolist(),
                    batch["texts"],
                    predictions,
                ):
                    writer.writerow(
                        {
                            "uid": uid,
                            "source_row": str(source_row),
                            "target": target,
                            "prediction": prediction,
                        }
                    )
                    rows_predicted += 1

                output_file.flush()
                if args.print_every and batch_index % args.print_every == 0:
                    print(f"predicted rows: {rows_predicted}", flush=True)

    print(f"rows predicted: {rows_predicted}")
    print(f"output        : {args.output}")


if __name__ == "__main__":
    main()
