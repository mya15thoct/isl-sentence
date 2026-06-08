"""Evaluate SignPose2Text predictions without external metric packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(
    references: list[list[str]],
    predictions: list[list[str]],
    max_order: int = 4,
) -> dict[str, float]:
    matches = [0] * max_order
    possible = [0] * max_order
    ref_len = 0
    pred_len = 0

    for ref, pred in zip(references, predictions):
        ref_len += len(ref)
        pred_len += len(pred)
        for order in range(1, max_order + 1):
            ref_counts = ngrams(ref, order)
            pred_counts = ngrams(pred, order)
            overlap = pred_counts & ref_counts
            matches[order - 1] += sum(overlap.values())
            possible[order - 1] += sum(pred_counts.values())

    precisions = [
        (matches[i] + 1.0) / (possible[i] + 1.0)
        for i in range(max_order)
    ]
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    if pred_len == 0:
        brevity = 0.0
    elif pred_len > ref_len:
        brevity = 1.0
    else:
        brevity = math.exp(1.0 - ref_len / max(pred_len, 1))

    bleu = brevity * geo_mean
    return {
        "bleu": bleu,
        **{f"bleu_p{i+1}": precisions[i] for i in range(max_order)},
        "brevity_penalty": brevity,
        "ref_len": float(ref_len),
        "pred_len": float(pred_len),
    }


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        cur = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = cur
    return prev[-1]


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_score(ref: list[str], pred: list[str]) -> float:
    if not ref or not pred:
        return 0.0
    lcs = lcs_length(ref, pred)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def evaluate_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    references = [tokenize(row.get("target", "")) for row in rows]
    predictions = [tokenize(row.get("prediction", "")) for row in rows]
    bleu = corpus_bleu(references, predictions)

    total_ref_words = sum(len(ref) for ref in references)
    total_edit = sum(edit_distance(ref, pred) for ref, pred in zip(references, predictions))
    exact = sum(
        normalize_text(row.get("target", "")) == normalize_text(row.get("prediction", ""))
        for row in rows
    )
    nonempty = sum(bool(pred) for pred in predictions)
    rouge_l = sum(rouge_l_score(ref, pred) for ref, pred in zip(references, predictions))
    pred_lengths = [len(pred) for pred in predictions]
    ref_lengths = [len(ref) for ref in references]

    metrics = {
        "rows": float(len(rows)),
        "exact_match": exact / max(len(rows), 1),
        "nonempty_prediction": nonempty / max(len(rows), 1),
        "wer": total_edit / max(total_ref_words, 1),
        "rouge_l": rouge_l / max(len(rows), 1),
        "avg_ref_words": sum(ref_lengths) / max(len(ref_lengths), 1),
        "avg_pred_words": sum(pred_lengths) / max(len(pred_lengths), 1),
    }
    metrics.update(bleu)
    return metrics


def write_json(path: Path, data: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_error_sample(path: Path, rows: list[dict[str, str]], limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scored: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        ref = tokenize(row.get("target", ""))
        pred = tokenize(row.get("prediction", ""))
        scored.append((edit_distance(ref, pred), row))
    scored.sort(key=lambda item: item[0], reverse=True)

    fields = ["uid", "source_row", "target", "prediction", "word_edit_distance"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for distance, row in scored[:limit]:
            writer.writerow(
                {
                    "uid": row.get("uid", ""),
                    "source_row": row.get("source_row", ""),
                    "target": row.get("target", ""),
                    "prediction": row.get("prediction", ""),
                    "word_edit_distance": str(distance),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--error-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_predictions(args.predictions)
    metrics = evaluate_rows(rows)
    write_json(args.output_json, metrics)
    if args.error_output:
        write_error_sample(args.error_output, rows, args.error_limit)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"metrics saved: {args.output_json}")
    if args.error_output:
        print(f"errors saved : {args.error_output}")


if __name__ == "__main__":
    main()
