"""Report per-seed mean+/-std AND the ensemble for a set of seed checkpoints.

Give it the N seed checkpoints of the SAME recipe (e.g. abl_ema50 seeds 42/43/44).
It encodes the manifest once per checkpoint, then reports, for every pool size and
re-ranking method:

  * per-seed  : mean +/- std of the individual seeds (robustness / honest single
                model number for the paper)
  * ensemble  : metrics of the averaged similarity matrix (the combined model -
                NOT "pick the best seed", which would be cherry-picking)

Usage:
    python -m src.retrieval.report_ensemble \
        --manifest .../isign_retrieval_videosplit_test.csv \
        --checkpoints .../abl_ema50/checkpoint_ema.pt \
                      .../abl_ema50_s43/checkpoint_ema.pt \
                      .../abl_ema50_s44/checkpoint_ema.pt \
        --pool-sizes 0 2000 1000 --out .../eval/seeds_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.dataset import RetrievalDataset, collate_retrieval
from src.retrieval.evaluate_rerank import (
    average_dicts,
    build_model_from_checkpoint,
    encode_manifest,
    evaluate_pool,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[0])
    parser.add_argument("--pool-seeds", type=int, default=3)
    parser.add_argument("--dsl-temp", type=float, default=100.0)
    parser.add_argument("--sinkhorn-temp", type=float, default=20.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, help="optional JSON dump")
    return parser.parse_args()


def eval_all_pools(sim: torch.Tensor, group_ids: torch.Tensor, args: argparse.Namespace) -> dict:
    """Replicate evaluate_rerank's multi-pool loop for one similarity matrix."""
    n = sim.size(0)
    out: dict[str, dict] = {}
    for pool_size in args.pool_sizes:
        if pool_size <= 0 or pool_size >= n:
            out[f"full({n})"] = evaluate_pool(
                sim, group_ids, None, args.dsl_temp, args.sinkhorn_temp, args.sinkhorn_iters
            )
        else:
            runs = []
            for seed in range(args.pool_seeds):
                generator = torch.Generator().manual_seed(seed)
                indices = torch.randperm(n, generator=generator)[:pool_size]
                runs.append(evaluate_pool(
                    sim, group_ids, indices, args.dsl_temp, args.sinkhorn_temp, args.sinkhorn_iters
                ))
            out[f"{pool_size}(avg{args.pool_seeds})"] = average_dicts(runs)
    return out


def reduce_mean_std(dicts: list):
    """Recursively turn a list of identically-shaped metric dicts into a dict
    whose leaves are {"mean": .., "std": ..} (population std)."""
    first = dicts[0]
    if isinstance(first, dict):
        return {key: reduce_mean_std([d[key] for d in dicts]) for key in first}
    values = [float(d) for d in dicts]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return {"mean": mean, "std": std}


def fmt_seed(block: dict, scheme: str) -> str:
    # block[scheme][direction][exact|redundancy][metric] -> {"mean","std"}
    e = block[scheme]["v2t"]["exact"]
    t = block[scheme]["t2v"]["exact"]
    def ms(d, m):
        return f"{d[m]['mean']:.3f}±{d[m]['std']:.3f}"
    return (f"v2t R@1 {ms(e,'r1')} R@5 {ms(e,'r5')} R@10 {ms(e,'r10')} "
            f"med {e['median_rank']['mean']:.1f} | t2v R@10 {ms(t,'r10')}")


def fmt_ens(block: dict, scheme: str) -> str:
    e = block[scheme]["v2t"]["exact"]
    t = block[scheme]["t2v"]["exact"]
    return (f"v2t R@1 {e['r1']:.3f} R@5 {e['r5']:.3f} R@10 {e['r10']:.3f} "
            f"med {e['median_rank']:.0f} | t2v R@10 {t['r10']:.3f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = RetrievalDataset(
        manifest=args.manifest,
        text_column=args.text_column,
        keypoint_column=args.keypoint_column,
        max_frames=args.max_frames,
        sample_mode="uniform",
        limit=args.limit,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_retrieval,
        pin_memory=device.type == "cuda",
    )
    print(f"rows: {len(dataset)}  seeds: {len(args.checkpoints)}")

    sims: list[torch.Tensor] = []
    group_ids: torch.Tensor | None = None
    for ckpt in args.checkpoints:
        print(f"encoding with {ckpt} ...", flush=True)
        model = build_model_from_checkpoint(ckpt, device)
        video, text, groups = encode_manifest(model, loader, device)
        sims.append(video @ text.T)
        group_ids = groups
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert group_ids is not None

    # per-seed: evaluate each seed's matrix, then mean+/-std across seeds
    per_seed_runs = [eval_all_pools(sim, group_ids, args) for sim in sims]
    pools = list(per_seed_runs[0].keys())
    per_seed = {pool: reduce_mean_std([r[pool] for r in per_seed_runs]) for pool in pools}

    # ensemble: average the similarity matrices, evaluate once
    ensemble_sim = torch.stack(sims, dim=0).mean(dim=0)
    ensemble = eval_all_pools(ensemble_sim, group_ids, args)

    schemes = list(per_seed_runs[0][pools[0]].keys())  # cosine / dsl / sinkhorn
    for pool in pools:
        print(f"\npool={pool}")
        for scheme in schemes:
            print(f"  [{scheme}]")
            print(f"    per-seed (n={len(sims)})  {fmt_seed(per_seed[pool], scheme)}")
            print(f"    ensemble              {fmt_ens(ensemble[pool], scheme)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"per_seed": per_seed, "ensemble": ensemble}, indent=2), encoding="utf-8")
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
