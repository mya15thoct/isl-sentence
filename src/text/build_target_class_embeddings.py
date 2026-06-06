"""
Build one text embedding per target sentence class.

The target set has many videos but only 98 sentence classes. For gloss-aware
fine-tuning, the classifier needs a (num_classes, embedding_dim) matrix aligned
with label_id, not one embedding per video.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="target_class_minilm")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--text-mode",
        choices=["sentence", "gloss", "sentence-gloss", "gloss-sentence"],
        default="sentence-gloss",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def class_text(row: dict[str, str], mode: str) -> str:
    sentence = (row.get("sentence") or row.get("label_text") or "").strip()
    gloss = (row.get("gloss") or "").strip()

    if mode == "sentence":
        return sentence
    if mode == "gloss":
        return gloss or sentence
    if mode == "sentence-gloss":
        return f"{sentence}. {gloss}" if gloss else sentence
    if mode == "gloss-sentence":
        return f"{gloss}. {sentence}" if gloss else sentence
    raise ValueError(f"Unknown text mode: {mode}")


def collect_classes(rows: list[dict[str, str]], text_mode: str) -> list[dict[str, str]]:
    by_id: dict[int, dict[str, str]] = {}
    for row in rows:
        label_id = int(row["label_id"])
        by_id.setdefault(label_id, row)

    label_ids = sorted(by_id)
    expected = list(range(len(label_ids)))
    if label_ids != expected:
        raise SystemExit(f"label_id must be contiguous 0..N-1, got first ids={label_ids[:10]}")

    output: list[dict[str, str]] = []
    for label_id in label_ids:
        row = by_id[label_id]
        text = class_text(row, text_mode)
        if not text:
            raise SystemExit(f"Empty class text for label_id={label_id}")
        output.append(
            {
                "label_id": str(label_id),
                "label_text": (row.get("label_text") or row.get("sentence") or "").strip(),
                "sentence": (row.get("sentence") or "").strip(),
                "gloss": (row.get("gloss") or "").strip(),
                "embedding_text": text,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label_id", "label_text", "sentence", "gloss", "embedding_text"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install with:\n"
            "  pip install sentence-transformers\n"
        ) from exc

    rows = read_manifest(args.manifest)
    classes = collect_classes(rows, args.text_mode)
    texts = [row["embedding_text"] for row in classes]

    print(f"Classes to encode: {len(texts)}")
    print(f"Text mode        : {args.text_mode}")
    print(f"Model            : {args.model}")

    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=not args.no_normalize,
        show_progress_bar=True,
    ).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.output_dir / f"{args.name}.npy"
    index_path = args.output_dir / f"{args.name}_index.csv"
    config_path = args.output_dir / f"{args.name}_config.json"

    np.save(emb_path, embeddings)
    write_csv(index_path, classes)
    config = {
        "manifest": str(args.manifest),
        "model": args.model,
        "text_mode": args.text_mode,
        "embedding_shape": list(embeddings.shape),
        "normalized": not args.no_normalize,
        "index_csv": str(index_path),
        "embeddings_npy": str(emb_path),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Embeddings saved : {emb_path}")
    print(f"Index saved      : {index_path}")
    print(f"Config saved     : {config_path}")


if __name__ == "__main__":
    main()
