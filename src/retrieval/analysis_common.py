"""Shared helpers for offline analysis over embedding dumps.

A "dump" is the .pt file written by ``evaluate_rerank --dump-embeddings``:
per-checkpoint fp16 video/text embeddings plus group ids, captions, uids and
(capped) frame lengths for every row of the evaluated manifest. All analysis
scripts (bootstrap CIs, re-rank sweeps, error analysis) start from a dump so
they never need the GPU or the keypoint files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.evaluate_rerank import dsl_rerank, sinkhorn_rerank

RERANK_METHODS = ("cosine", "dsl", "sinkhorn")


def load_dump(path: Path) -> dict:
    dump = torch.load(path, map_location="cpu", weights_only=False)
    required = {"video_embeddings", "text_embeddings", "group_ids", "captions"}
    missing = required - set(dump)
    if missing:
        raise ValueError(f"{path} is not an embedding dump (missing keys: {sorted(missing)})")
    return dump


def similarity_from_dump(dump: dict) -> torch.Tensor:
    """(N, N) cosine similarity; multi-checkpoint dumps are late-fusion averaged
    exactly as in evaluate_rerank (mean of per-checkpoint similarity matrices)."""
    sims = [
        video.float() @ text.float().T
        for video, text in zip(dump["video_embeddings"], dump["text_embeddings"])
    ]
    return sims[0] if len(sims) == 1 else torch.stack(sims).mean(dim=0)


def rerank(
    sim: torch.Tensor,
    method: str,
    dsl_temp: float = 100.0,
    sinkhorn_temp: float = 20.0,
    sinkhorn_iters: int = 4,
) -> torch.Tensor:
    if method == "cosine":
        return sim
    if method == "dsl":
        return dsl_rerank(sim, dsl_temp)
    if method == "sinkhorn":
        return sinkhorn_rerank(sim, sinkhorn_temp, sinkhorn_iters)
    raise ValueError(f"unknown re-rank method {method!r} (choose from {RERANK_METHODS})")


def per_query_ranks(scores: torch.Tensor, group_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-query rank vectors for a (Q, C) score matrix with queries in rows.

    ``exact``      - rank of the paired (diagonal) candidate.
    ``redundancy`` - rank of the best candidate sharing the query's caption group.
    """
    target = scores.diag().unsqueeze(1)
    exact = (scores > target).sum(dim=1) + 1

    relevant = group_ids.view(-1, 1) == group_ids.view(1, -1)
    neg_inf = torch.finfo(scores.dtype).min
    best_relevant = scores.masked_fill(~relevant, neg_inf).max(dim=1).values
    redundancy = (scores > best_relevant.unsqueeze(1)).sum(dim=1) + 1

    return {"exact": exact, "redundancy": redundancy}


def recall_at(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean())
