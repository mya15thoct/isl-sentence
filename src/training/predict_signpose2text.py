"""Generate SignPose2Text predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

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
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config") or {}
    return dict(config)


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["uid", "source_row", "target", "prediction"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_config(checkpoint)

    text_model = args.text_model or config.get("text_model", "t5-small")
    text_column = args.text_column or config.get("text_column", "canonical_text")
    keypoint_column = args.keypoint_column or config.get("keypoint_column", "keypoint_path")
    max_frames = args.max_frames or int(config.get("max_frames", 512))
    max_target_tokens = args.max_target_tokens or int(config.get("max_target_tokens", 96))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = ConformerT5SignPose2Text(
        text_model_name=text_model,
        keypoint_model_dim=int(config.get("keypoint_model_dim", 256)),
        keypoint_layers=int(config.get("keypoint_layers", 4)),
        keypoint_heads=int(config.get("keypoint_heads", 4)),
        downsample_stride=int(config.get("downsample_stride", 4)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = SignPose2TextDataset(
        manifest=args.manifest,
        text_column=text_column,
        keypoint_column=keypoint_column,
        max_frames=max_frames,
        sample_mode="uniform",
        limit=args.limit,
    )
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

    output_rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch in loader:
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
                output_rows.append(
                    {
                        "uid": uid,
                        "source_row": str(source_row),
                        "target": target,
                        "prediction": prediction,
                    }
                )

    write_predictions(args.output, output_rows)
    print(f"rows predicted: {len(output_rows)}")
    print(f"output        : {args.output}")


if __name__ == "__main__":
    main()
