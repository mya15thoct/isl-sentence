"""Offline retrieval evaluation with re-ranking, ensembling and pool sizes.

Takes one or more trained checkpoints (each stores its own model config, so
mixed architectures / text models are fine) and reports retrieval metrics on a
manifest under several scoring schemes:

  * cosine    - plain cosine similarity (same as training-time validation)
  * dsl       - Dual-Softmax re-ranking (CAMoE / CLIP4Clip inference trick):
                each candidate's score is weighted by a softmax over all
                queries, demoting "hub" candidates that are close to everything

When several checkpoints are given their cosine similarity matrices are
averaged (late-fusion ensemble) before ranking.

Metrics are reported for the full candidate pool and, optionally, for random
sub-pools (e.g. 1000 / 2000) averaged over several seeds - this matches the
gallery sizes used by How2Sign-style retrieval papers and makes cross-dataset
comparison fair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.retrieval import PoseTextRetrievalModel
from src.models.text_encoder import DEFAULT_TEXT_MODEL
from src.retrieval.dataset import RetrievalDataset, collate_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--keypoint-column", default="keypoint_path")
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=[0],
        help="candidate pool sizes; 0 = full pool",
    )
    parser.add_argument("--pool-seeds", type=int, default=3, help="random sub-pools per size")
    parser.add_argument("--dsl-temp", type=float, default=100.0, help="sharpness of the DSL prior")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, help="optional JSON dump of all metrics")
    return parser.parse_args()


def build_model_from_checkpoint(path: Path, device: torch.device) -> PoseTextRetrievalModel:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    model = PoseTextRetrievalModel(
        embedding_dim=int(cfg.get("embedding_dim", 384)),
        pose_model_dim=int(cfg.get("pose_model_dim", 256)),
        pose_layers=int(cfg.get("pose_layers", 4)),
        pose_heads=int(cfg.get("pose_heads", 4)),
        downsample_stride=int(cfg.get("downsample_stride", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        hand_aware=bool(cfg.get("hand_aware", False)),
        context_parts=tuple(cfg.get("context_parts", ("pose", "face"))),
        text_model_name=str(cfg.get("text_model", DEFAULT_TEXT_MODEL)),
        max_text_length=int(cfg.get("max_text_length", 64)),
    )
    model.load_state_dict(state["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def encode_manifest(
    model: PoseTextRetrievalModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    video_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    group_chunks: list[torch.Tensor] = []
    for batch in loader:
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        tokens = model.text_encoder.tokenize(batch["texts"])
        tokens = {key: value.to(device) for key, value in tokens.items()}
        video, _, _ = model.encode_pose(keypoints, lengths)
        text = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
        video_chunks.append(video.float().cpu())
        text_chunks.append(text.float().cpu())
        group_chunks.append(batch["group_ids"])
    video = F.normalize(torch.cat(video_chunks), dim=-1, eps=1e-6)
    text = F.normalize(torch.cat(text_chunks), dim=-1, eps=1e-6)
    groups = torch.cat(group_chunks)
    return video, text, groups


def dsl_rerank(sim: torch.Tensor, temp: float) -> torch.Tensor:
    """Dual-Softmax: scale scores by a softmax prior over the QUERY axis.

    ``sim`` is (Q, C) with queries in rows. A candidate that scores high
    against many queries gets a diluted prior, suppressing hub candidates.
    """
    prior = torch.softmax(sim * temp, dim=0)
    return sim * prior


def rank_summary(ranks: torch.Tensor) -> dict[str, float]:
    ranks = ranks.float()
    return {
        "r1": float((ranks <= 1).float().mean()),
        "r5": float((ranks <= 5).float().mean()),
        "r10": float((ranks <= 10).float().mean()),
        "median_rank": float(ranks.median()),
    }


def metrics_for_scores(scores: torch.Tensor, group_ids: torch.Tensor) -> dict[str, dict[str, float]]:
    """Exact (diagonal) and redundancy-aware ranks for a (N, N) score matrix."""
    target = scores.diag().unsqueeze(1)
    exact = (scores > target).sum(dim=1) + 1

    relevant = group_ids.view(-1, 1) == group_ids.view(1, -1)
    neg_inf = torch.finfo(scores.dtype).min
    best_relevant = scores.masked_fill(~relevant, neg_inf).max(dim=1).values
    redundancy = (scores > best_relevant.unsqueeze(1)).sum(dim=1) + 1

    return {"exact": rank_summary(exact), "redundancy": rank_summary(redundancy)}


def evaluate_pool(
    sim: torch.Tensor,
    group_ids: torch.Tensor,
    indices: torch.Tensor | None,
    dsl_temp: float,
) -> dict[str, dict]:
    if indices is not None:
        sim = sim[indices][:, indices]
        group_ids = group_ids[indices]
    results: dict[str, dict] = {}
    for method, v2t_scores in (
        ("cosine", sim),
        ("dsl", dsl_rerank(sim, dsl_temp)),
    ):
        t2v_scores = dsl_rerank(sim.T, dsl_temp) if method == "dsl" else sim.T
        results[method] = {
            "v2t": metrics_for_scores(v2t_scores, group_ids),
            "t2v": metrics_for_scores(t2v_scores, group_ids),
        }
    return results


def average_dicts(dicts: list[dict]) -> dict:
    """Recursively average a list of identically-shaped nested metric dicts."""
    first = dicts[0]
    if isinstance(first, dict):
        return {key: average_dicts([d[key] for d in dicts]) for key in first}
    return sum(dicts) / len(dicts)


def format_block(name: str, block: dict) -> str:
    v2t, t2v = block["v2t"]["exact"], block["t2v"]["exact"]
    rv2t = block["v2t"]["redundancy"]
    return (
        f"  {name:<8} v2t r1/r5/r10={v2t['r1']:.3f}/{v2t['r5']:.3f}/{v2t['r10']:.3f} "
        f"med={v2t['median_rank']:.0f} | t2v r10={t2v['r10']:.3f} | "
        f"rv2t r10={rv2t['r10']:.3f}"
    )


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
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_retrieval,
        pin_memory=device.type == "cuda",
    )
    print(f"rows: {len(dataset)}  checkpoints: {len(args.checkpoints)}")

    sim_sum: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None
    for ckpt in args.checkpoints:
        print(f"encoding with {ckpt} ...", flush=True)
        model = build_model_from_checkpoint(ckpt, device)
        video, text, groups = encode_manifest(model, loader, device)
        sim = video @ text.T
        sim_sum = sim if sim_sum is None else sim_sum + sim
        group_ids = groups
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert sim_sum is not None and group_ids is not None
    sim = sim_sum / len(args.checkpoints)

    n = sim.size(0)
    all_results: dict[str, dict] = {}
    for pool_size in args.pool_sizes:
        if pool_size <= 0 or pool_size >= n:
            key = f"pool=full({n})"
            all_results[key] = evaluate_pool(sim, group_ids, None, args.dsl_temp)
        else:
            runs = []
            for seed in range(args.pool_seeds):
                generator = torch.Generator().manual_seed(seed)
                indices = torch.randperm(n, generator=generator)[:pool_size]
                runs.append(evaluate_pool(sim, group_ids, indices, args.dsl_temp))
            key = f"pool={pool_size}(avg {args.pool_seeds} seeds)"
            all_results[key] = average_dicts(runs)

        print(f"\n{key}")
        for method, block in all_results[key].items():
            print(format_block(method, block))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
