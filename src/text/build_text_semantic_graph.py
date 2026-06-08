"""Build a nearest-neighbor semantic graph over cached text embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--self-weight", type=float, default=1.0)
    parser.add_argument("--neighbor-power", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_index(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_embeddings(path: Path, limit: int | None) -> np.ndarray:
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise SystemExit(f"Expected a 2D embedding array, got shape {embeddings.shape}.")
    if limit is not None:
        embeddings = embeddings[:limit]
    embeddings = np.asarray(embeddings, dtype=np.float32).copy()
    embeddings = np.nan_to_num(embeddings, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norm, 1e-6, None)
    return embeddings.astype(np.float32, copy=False)


def faiss_search(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError:
        return None

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    scores, ids = index.search(embeddings, k)
    return scores.astype(np.float32), ids.astype(np.int64)


def torch_search(
    embeddings: np.ndarray,
    k: int,
    device_name: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    bank = torch.from_numpy(embeddings).to(device)
    bank = F.normalize(bank.float(), dim=-1, eps=1e-6)
    all_scores: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    start_time = time.time()

    for start in range(0, bank.size(0), chunk_size):
        end = min(start + chunk_size, bank.size(0))
        scores = bank[start:end] @ bank.T
        top_scores, top_ids = torch.topk(scores, k=k, dim=1)
        all_scores.append(top_scores.detach().cpu().numpy().astype(np.float32))
        all_ids.append(top_ids.detach().cpu().numpy().astype(np.int64))
        done = end
        if done % max(chunk_size * 10, 1) == 0 or done == bank.size(0):
            elapsed = max(time.time() - start_time, 1e-6)
            print(f"searched rows={done}/{bank.size(0)} rows_per_sec={done / elapsed:.1f}", flush=True)

    return np.concatenate(all_scores, axis=0), np.concatenate(all_ids, axis=0)


def build_graph(
    scores: np.ndarray,
    ids: np.ndarray,
    top_k: int,
    threshold: float,
    self_weight: float,
    neighbor_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = ids.shape[0]
    positive_ids = np.full((n, top_k + 1), -1, dtype=np.int64)
    positive_weights = np.zeros((n, top_k + 1), dtype=np.float32)
    positive_scores = np.zeros((n, top_k + 1), dtype=np.float32)

    positive_ids[:, 0] = np.arange(n, dtype=np.int64)
    positive_weights[:, 0] = np.float32(self_weight)
    positive_scores[:, 0] = 1.0

    counts = np.ones(n, dtype=np.int32)
    for row_index in range(n):
        write_at = 1
        seen = {row_index}
        for score, neighbor_id in zip(scores[row_index], ids[row_index]):
            neighbor = int(neighbor_id)
            if neighbor in seen or neighbor < 0:
                continue
            seen.add(neighbor)
            if float(score) < threshold:
                continue
            if write_at >= top_k + 1:
                break
            positive_ids[row_index, write_at] = neighbor
            positive_scores[row_index, write_at] = np.float32(score)
            positive_weights[row_index, write_at] = np.float32(max(float(score), 0.0) ** neighbor_power)
            write_at += 1
        counts[row_index] = write_at

    return positive_ids, positive_weights, positive_scores


def write_report(path: Path, index_rows: list[dict[str, str]], ids: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "embedding_id",
        "text",
        "positive_count",
        "neighbors",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_index in range(ids.shape[0]):
            row = index_rows[row_index] if row_index < len(index_rows) else {}
            neighbors: list[str] = []
            for neighbor_id, score in zip(ids[row_index, 1:], scores[row_index, 1:]):
                if int(neighbor_id) < 0:
                    continue
                neighbor_row = index_rows[int(neighbor_id)] if int(neighbor_id) < len(index_rows) else {}
                text = neighbor_row.get("text", "")
                neighbors.append(f"{int(neighbor_id)}:{float(score):.3f}:{text[:80]}")
            writer.writerow(
                {
                    "embedding_id": str(row_index),
                    "text": row.get("text", ""),
                    "positive_count": str(int((ids[row_index] >= 0).sum())),
                    "neighbors": " | ".join(neighbors),
                }
            )


def main() -> None:
    args = parse_args()
    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive.")

    embeddings = normalize_embeddings(args.embeddings, args.limit)
    n = embeddings.shape[0]
    search_k = min(max(args.top_k + 1, 1), n)
    print(f"embeddings : {args.embeddings}")
    print(f"rows       : {n}")
    print(f"dim        : {embeddings.shape[1]}")
    print(f"top_k      : {args.top_k}")
    print(f"threshold  : {args.threshold}")

    result = None if args.no_faiss else faiss_search(embeddings, search_k)
    if result is None:
        print(f"search     : torch/{args.device}")
        scores, ids = torch_search(embeddings, search_k, args.device, args.chunk_size)
    else:
        print("search     : faiss IndexFlatIP")
        scores, ids = result

    positive_ids, positive_weights, positive_scores = build_graph(
        scores=scores,
        ids=ids,
        top_k=args.top_k,
        threshold=args.threshold,
        self_weight=args.self_weight,
        neighbor_power=args.neighbor_power,
    )
    counts = (positive_ids >= 0).sum(axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positive_ids=positive_ids,
        positive_weights=positive_weights,
        positive_scores=positive_scores,
        threshold=np.array([args.threshold], dtype=np.float32),
        top_k=np.array([args.top_k], dtype=np.int32),
    )

    metadata = {
        "embeddings": str(args.embeddings),
        "rows": int(n),
        "dim": int(embeddings.shape[1]),
        "top_k": int(args.top_k),
        "threshold": float(args.threshold),
        "self_weight": float(args.self_weight),
        "neighbor_power": float(args.neighbor_power),
        "mean_positive_count": float(counts.mean()),
        "median_positive_count": float(np.median(counts)),
        "max_positive_count": int(counts.max()),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.report_csv is not None:
        index_rows = read_index(args.index_csv)
        write_report(args.report_csv, index_rows, positive_ids, positive_scores)

    print(f"mean positives : {counts.mean():.2f}")
    print(f"median positives: {float(np.median(counts)):.2f}")
    print(f"max positives  : {counts.max()}")
    print(f"graph saved    : {args.output}")
    print(f"metadata saved : {metadata_path}")
    if args.report_csv is not None:
        print(f"report saved   : {args.report_csv}")


if __name__ == "__main__":
    main()
