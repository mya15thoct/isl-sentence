"""
Build text embedding caches from one or more manifest CSV files.

This is used for auxiliary semantic supervision in SignPose2Text experiments:
  text -> MiniLM/Sentence-BERT -> fixed text embedding

Example:
  python src/text/embeddings.py \
      --manifest data/subsets/isign_examples.csv \
                 data/subsets/isign_descriptions.csv \
                 data/subsets/isign_general_sentences.csv \
      --output-dir /mnt/ngan/ISL-Sequences/text_embeddings \
      --name isign_sentences_minilm \
      --model sentence-transformers/all-MiniLM-L6-v2 \
      --batch-size 256
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PASSTHROUGH_FIELDS = [
    "relation_type",
    "video_uid",
    "video_category",
    "source_text",
    "relation_weight",
    "pair_id",
    "sentence_uid",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(manifest_paths: list[Path], text_column: str) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []

    for manifest_path in manifest_paths:
        rows = read_manifest(manifest_path)
        for row_index, row in enumerate(rows):
            text = row.get(text_column, "").strip()
            if not text:
                continue

            output_rows.append(
                {
                    "embedding_id": str(len(output_rows)),
                    "source_manifest": str(manifest_path),
                    "source_row": str(row_index),
                    "uid": row.get("uid", "").strip(),
                    "text": text,
                    "category": row.get("category", "").strip(),
                    "word": row.get("word", "").strip(),
                    "word_id": row.get("word_id", "").strip(),
                    "keypoint_path": row.get("keypoint_path", "").strip(),
                    "num_frames": row.get("num_frames", "").strip(),
                    "extract_status": row.get("extract_status", "").strip(),
                    "quality_flag": row.get("quality_flag", "").strip(),
                    **{field: row.get(field, "").strip() for field in PASSTHROUGH_FIELDS},
                }
            )

    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="text_embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sentence-transformers device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install with:\n"
            "  pip install sentence-transformers\n"
        ) from exc

    rows = load_rows(args.manifest, args.text_column)
    if not rows:
        raise SystemExit("No non-empty text rows found.")

    texts = [row["text"] for row in rows]
    print(f"Rows to encode : {len(texts)}")
    print(f"Model          : {args.model}")

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
    fieldnames = [
            "embedding_id",
            "source_manifest",
            "source_row",
            "uid",
            "text",
            "category",
            "word",
            "word_id",
            "keypoint_path",
            "num_frames",
            "extract_status",
            "quality_flag",
    ]
    for field in PASSTHROUGH_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(index_path, rows, fieldnames)

    config = {
        "model": args.model,
        "embedding_shape": list(embeddings.shape),
        "normalized": not args.no_normalize,
        "manifests": [str(p) for p in args.manifest],
        "text_column": args.text_column,
        "index_csv": str(index_path),
        "embeddings_npy": str(emb_path),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Embeddings saved : {emb_path}")
    print(f"Index saved      : {index_path}")
    print(f"Config saved     : {config_path}")


if __name__ == "__main__":
    main()
