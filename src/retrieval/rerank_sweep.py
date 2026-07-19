"""Re-ranking sensitivity + cost analysis (review Issue 8).

Sweeps the Sinkhorn hyper-parameters (logit scale x iterations) and the DSL
temperature over an embedding dump, at several pool sizes, and reports
R@1/R@5/R@10 (V2T and T2V, exact scoring) plus the wall-clock runtime of each
re-ranking call. Shows whether the Sinkhorn gain is robust or
hyperparameter/pool-size sensitive, and what the transductive step costs.

Input is a dump from ``evaluate_rerank --dump-embeddings``; runs on CPU.
Sub-pools use the same fixed seeds as evaluate_rerank (torch.randperm with
manual_seed 0..pool_seeds-1) so numbers line up with the main tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.analysis_common import load_dump, per_query_ranks, recall_at, similarity_from_dump
from src.retrieval.evaluate_rerank import dsl_rerank, sinkhorn_rerank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--sinkhorn-temps", type=float, nargs="+", default=[10.0, 20.0, 40.0])
    parser.add_argument("--sinkhorn-iters", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--dsl-temps", type=float, nargs="+", default=[50.0, 100.0, 200.0])
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[0, 2000, 1000],
                        help="0 = full pool")
    parser.add_argument("--pool-seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, help="optional JSON output")
    return parser.parse_args()


def timed_metrics(
    sim: torch.Tensor,
    group_ids: torch.Tensor,
    fn,
) -> tuple[dict[str, float], float]:
    """Apply re-ranker ``fn`` per direction, return metrics + re-rank seconds."""
    start = time.perf_counter()
    v2t_scores = fn(sim)
    t2v_scores = fn(sim.T)
    seconds = time.perf_counter() - start
    out: dict[str, float] = {}
    for direction, scores in (("v2t", v2t_scores), ("t2v", t2v_scores)):
        ranks = per_query_ranks(scores, group_ids)["exact"]
        for k in (1, 5, 10):
            out[f"{direction}_r{k}"] = recall_at(ranks, k)
        out[f"{direction}_median"] = float(ranks.float().median())
    return out, seconds


def sweep_pool(sim: torch.Tensor, group_ids: torch.Tensor, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    configs: list[tuple[str, dict, object]] = [("cosine", {}, lambda s: s)]
    for temp in args.dsl_temps:
        configs.append((f"dsl", {"temp": temp}, lambda s, t=temp: dsl_rerank(s, t)))
    for temp in args.sinkhorn_temps:
        for iters in args.sinkhorn_iters:
            configs.append(
                (f"sinkhorn", {"temp": temp, "iters": iters},
                 lambda s, t=temp, i=iters: sinkhorn_rerank(s, t, i))
            )
    for method, params, fn in configs:
        metrics, seconds = timed_metrics(sim, group_ids, fn)
        rows.append({"method": method, **params, **metrics, "rerank_seconds": round(seconds, 3)})
    return rows


def print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n### {title}")
    print("| method | temp | iters | v2t R@1 | v2t R@5 | v2t R@10 | v2t med | t2v R@10 | rerank s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['method']} | {r.get('temp', '-')} | {r.get('iters', '-')} "
            f"| {r['v2t_r1']:.3f} | {r['v2t_r5']:.3f} | {r['v2t_r10']:.3f} "
            f"| {r['v2t_median']:.0f} | {r['t2v_r10']:.3f} | {r['rerank_seconds']} |"
        )


def main() -> None:
    args = parse_args()
    dump = load_dump(args.dump)
    sim = similarity_from_dump(dump)
    group_ids = dump["group_ids"]
    n = sim.size(0)
    print(f"dump: {args.dump}  queries: {n}")

    results: dict[str, list[dict]] = {}
    for pool_size in args.pool_sizes:
        if pool_size <= 0 or pool_size >= n:
            key = f"pool=full({n})"
            rows = sweep_pool(sim, group_ids, args)
        else:
            key = f"pool={pool_size}(avg {args.pool_seeds} seeds)"
            per_seed: list[list[dict]] = []
            for seed in range(args.pool_seeds):
                generator = torch.Generator().manual_seed(seed)
                indices = torch.randperm(n, generator=generator)[:pool_size]
                per_seed.append(sweep_pool(sim[indices][:, indices], group_ids[indices], args))
            rows = []
            for cfg_rows in zip(*per_seed):
                averaged = dict(cfg_rows[0])
                for field in averaged:
                    if field not in ("method", "temp", "iters"):
                        averaged[field] = round(sum(r[field] for r in cfg_rows) / len(cfg_rows), 4)
                rows.append(averaged)
        results[key] = rows
        print_rows(key, rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
