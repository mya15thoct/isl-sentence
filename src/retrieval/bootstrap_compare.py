"""Paired bootstrap significance test for retrieval comparisons (review Issue 4).

Answers "is the R@K difference between system A and system B statistically
significant, or seed/query noise?" by resampling TEST queries with replacement
(the candidate pool stays fixed - standard paired bootstrap for retrieval) and
reporting the observed delta, its 95% percentile CI and a two-sided p-value for
R@1/R@5/R@10 and median rank, in both directions (V2T, T2V).

Systems are embedding dumps written by ``evaluate_rerank --dump-embeddings``.
Two usage patterns:

  * two models, same re-ranker (e.g. hand-aware vs uniform fusion):
      --dump-a emb_ema50.pt --dump-b emb_no_handaware.pt --method-a sinkhorn
  * one model, two re-rankers (e.g. Sinkhorn vs DSL):
      --dump-a emb_ema50.pt --method-a sinkhorn --method-b dsl

Both dumps must come from the SAME manifest (row order included); this is
verified via captions and group ids before comparing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.analysis_common import (
    RERANK_METHODS,
    load_dump,
    per_query_ranks,
    rerank,
    similarity_from_dump,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-a", type=Path, required=True)
    parser.add_argument("--dump-b", type=Path, help="defaults to --dump-a (for method-vs-method comparisons)")
    parser.add_argument("--method-a", choices=RERANK_METHODS, default="sinkhorn")
    parser.add_argument("--method-b", choices=RERANK_METHODS, help="defaults to --method-a")
    parser.add_argument("--label-a", help="name for system A in the report")
    parser.add_argument("--label-b", help="name for system B in the report")
    parser.add_argument("--scoring", choices=("exact", "redundancy"), default="exact")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dsl-temp", type=float, default=100.0)
    parser.add_argument("--sinkhorn-temp", type=float, default=20.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=4)
    parser.add_argument("--out", type=Path, help="optional JSON output")
    return parser.parse_args()


def check_same_manifest(dump_a: dict, dump_b: dict) -> None:
    if dump_a["captions"] != dump_b["captions"] or not torch.equal(
        dump_a["group_ids"], dump_b["group_ids"]
    ):
        raise SystemExit(
            "dumps were built from different manifests (captions/groups differ); "
            "a paired test needs identical query sets in identical order"
        )


def paired_bootstrap(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    statistic: str,
    n_boot: int,
    seed: int,
    chunk: int = 500,
) -> dict[str, float]:
    """Bootstrap the difference (A - B) of a per-query statistic.

    ``statistic`` is "mean" (for R@K indicator vectors) or "median" (for ranks).
    """
    n = values_a.numel()
    generator = torch.Generator().manual_seed(seed)
    diffs: list[torch.Tensor] = []
    remaining = n_boot
    while remaining > 0:
        b = min(chunk, remaining)
        idx = torch.randint(0, n, (b, n), generator=generator)
        sampled_a, sampled_b = values_a[idx].float(), values_b[idx].float()
        if statistic == "mean":
            diffs.append(sampled_a.mean(dim=1) - sampled_b.mean(dim=1))
        else:
            diffs.append(sampled_a.median(dim=1).values - sampled_b.median(dim=1).values)
        remaining -= b
    delta = torch.cat(diffs)

    if statistic == "mean":
        observed = float(values_a.float().mean() - values_b.float().mean())
    else:
        observed = float(values_a.float().median() - values_b.float().median())
    lo, hi = (float(q) for q in torch.quantile(delta, torch.tensor([0.025, 0.975])))
    # Two-sided percentile p-value, floored at the bootstrap resolution.
    p = 2.0 * min(float((delta <= 0).float().mean()), float((delta >= 0).float().mean()))
    p = min(1.0, max(p, 1.0 / n_boot))
    return {"delta": observed, "ci_low": lo, "ci_high": hi, "p_value": p,
            "significant_95": bool(lo > 0 or hi < 0)}


def main() -> None:
    args = parse_args()
    method_b = args.method_b or args.method_a
    dump_b_path = args.dump_b or args.dump_a
    if dump_b_path == args.dump_a and method_b == args.method_a:
        raise SystemExit("A and B are identical: pass --dump-b and/or --method-b")

    label_a = args.label_a or f"{args.dump_a.stem}/{args.method_a}"
    label_b = args.label_b or f"{dump_b_path.stem}/{method_b}"

    dump_a = load_dump(args.dump_a)
    dump_b = dump_a if dump_b_path == args.dump_a else load_dump(dump_b_path)
    check_same_manifest(dump_a, dump_b)
    group_ids = dump_a["group_ids"]

    sim_a = similarity_from_dump(dump_a)
    sim_b = sim_a if dump_b is dump_a else similarity_from_dump(dump_b)
    rerank_kwargs = dict(
        dsl_temp=args.dsl_temp, sinkhorn_temp=args.sinkhorn_temp, sinkhorn_iters=args.sinkhorn_iters
    )

    print(f"A = {label_a}   B = {label_b}")
    print(f"queries: {sim_a.size(0)}   scoring: {args.scoring}   bootstrap: {args.n_boot} resamples\n")

    report: dict[str, dict] = {"a": label_a, "b": label_b, "scoring": args.scoring,
                               "n_boot": args.n_boot, "directions": {}}
    for direction in ("v2t", "t2v"):
        scores_a = rerank(sim_a if direction == "v2t" else sim_a.T, args.method_a, **rerank_kwargs)
        scores_b = rerank(sim_b if direction == "v2t" else sim_b.T, method_b, **rerank_kwargs)
        ranks_a = per_query_ranks(scores_a, group_ids)[args.scoring]
        ranks_b = per_query_ranks(scores_b, group_ids)[args.scoring]
        del scores_a, scores_b

        block: dict[str, dict] = {}
        for k in (1, 5, 10):
            block[f"r{k}"] = paired_bootstrap(
                (ranks_a <= k), (ranks_b <= k), "mean", args.n_boot, args.seed
            )
        block["median_rank"] = paired_bootstrap(ranks_a, ranks_b, "median", args.n_boot, args.seed)
        report["directions"][direction] = block

        print(f"[{direction}]")
        for name, stats in block.items():
            marker = "SIGNIFICANT" if stats["significant_95"] else "not significant"
            print(
                f"  d{name:<12} = {stats['delta']:+.4f}  "
                f"95% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
                f"p={stats['p_value']:.4g}  -> {marker}"
            )
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
