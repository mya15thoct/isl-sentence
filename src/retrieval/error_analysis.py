"""Qualitative + stratified error analysis for the retrieval model (review Issue 7).

From an embedding dump (``evaluate_rerank --dump-embeddings``) this produces the
"Error Analysis" material the paper needs:

  1. R@1/R@10/median stratified by caption length (words);
  2. ... by caption-group size (unique vs duplicated captions), under both
     exact and redundancy-aware scoring;
  3. ... by (capped) clip length in frames;
  4. hub analysis: which captions crowd the top-10 lists under cosine, and how
     Sinkhorn changes that (top hubs + share of all top-10 slots);
  5. queries fixed / broken by Sinkhorn (top-10 transitions vs cosine);
  6. the N worst V2T failures with their retrieved neighbours, before/after
     Sinkhorn - ready to paste into a qualitative table.

Writes a Markdown report (--report) and a JSON summary (--out). CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.analysis_common import load_dump, per_query_ranks, recall_at, rerank, similarity_from_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--sinkhorn-temp", type=float, default=20.0)
    parser.add_argument("--sinkhorn-iters", type=int, default=4)
    parser.add_argument("--length-bins", type=int, nargs="+", default=[4, 8, 15],
                        help="word-count bin edges; bins are <=e1, e1+1..e2, ..., >last")
    parser.add_argument("--frame-bins", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--top-failures", type=int, default=20)
    parser.add_argument("--neighbours", type=int, default=3)
    parser.add_argument("--hub-top", type=int, default=10)
    parser.add_argument("--manifest", type=Path,
                        help="optional: manifest CSV to join extra columns by uid")
    parser.add_argument("--join-columns", nargs="*", default=["category"],
                        help="manifest columns to stratify by (numeric -> quartiles); "
                             "missing columns are skipped with a warning")
    parser.add_argument("--report", type=Path, help="Markdown report path")
    parser.add_argument("--out", type=Path, help="JSON summary path")
    return parser.parse_args()


def manifest_strata_labels(
    manifest: Path, uids: list[str], column: str
) -> list[str] | None:
    """Per-query labels for a manifest column, joined by uid.

    Numeric columns are binned into quartiles; categorical values pass through.
    Returns None (with a warning) when the column is absent or uids don't match.
    """
    from src.retrieval.dataset import read_csv

    rows = read_csv(manifest)
    if not rows or column not in rows[0]:
        print(f"[join] column {column!r} not in manifest — skipped")
        return None
    by_uid = {row.get("uid", ""): row.get(column, "").strip() for row in rows}
    values = [by_uid.get(uid) for uid in uids]
    matched = sum(v is not None for v in values)
    if matched < 0.9 * len(uids):
        print(f"[join] only {matched}/{len(uids)} uids matched for {column!r} — skipped")
        return None
    present = [v for v in values if v]
    try:
        numbers = [float(v) for v in present]
    except ValueError:
        return [v if v else "(missing)" for v in values]
    qs = torch.quantile(torch.tensor(numbers), torch.tensor([0.25, 0.5, 0.75])).tolist()

    def bucket(v: str | None) -> str:
        if not v:
            return "(missing)"
        x = float(v)
        if x <= qs[0]:
            return f"Q1 (<= {qs[0]:.3g})"
        if x <= qs[1]:
            return f"Q2 (<= {qs[1]:.3g})"
        if x <= qs[2]:
            return f"Q3 (<= {qs[2]:.3g})"
        return f"Q4 (> {qs[2]:.3g})"

    return [bucket(v) for v in values]


def bin_label(value: int, edges: list[int]) -> str:
    lo = None
    for edge in edges:
        if value <= edge:
            return f"{(lo + 1) if lo is not None else 0}-{edge}"
        lo = edge
    return f">{edges[-1]}"


def stratify(
    ranks_by_method: dict[str, torch.Tensor],
    labels: list[str],
    order: list[str],
) -> list[dict]:
    rows = []
    labels_t = labels
    for bucket in order:
        idx = torch.tensor([i for i, lab in enumerate(labels_t) if lab == bucket], dtype=torch.long)
        if idx.numel() == 0:
            continue
        row: dict = {"bucket": bucket, "n": int(idx.numel())}
        for method, ranks in ranks_by_method.items():
            sub = ranks[idx]
            row[f"{method}_r1"] = round(recall_at(sub, 1), 3)
            row[f"{method}_r10"] = round(recall_at(sub, 10), 3)
            row[f"{method}_median"] = float(sub.float().median())
        rows.append(row)
    return rows


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "(empty)\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def hub_stats(scores: torch.Tensor, captions: list[str], top_n: int) -> dict:
    """How concentrated are the top-10 lists on a few 'hub' candidates?"""
    top10 = torch.topk(scores, k=min(10, scores.size(1)), dim=1).indices.flatten()
    counts = Counter(top10.tolist())
    total_slots = int(top10.numel())
    hub100 = sum(count for _, count in counts.most_common(100))
    return {
        "top_hubs": [
            {"caption": captions[idx][:70], "top10_appearances": count}
            for idx, count in counts.most_common(top_n)
        ],
        "distinct_candidates_in_top10": len(counts),
        "share_of_slots_taken_by_top100_hubs": round(hub100 / total_slots, 4),
    }


def main() -> None:
    args = parse_args()
    dump = load_dump(args.dump)
    captions: list[str] = dump["captions"]
    group_ids: torch.Tensor = dump["group_ids"]
    lengths: torch.Tensor = dump["lengths"]

    sim = similarity_from_dump(dump)
    scores = {
        "cosine": sim,
        "sinkhorn": rerank(sim, "sinkhorn", sinkhorn_temp=args.sinkhorn_temp,
                           sinkhorn_iters=args.sinkhorn_iters),
    }
    ranks = {m: per_query_ranks(s, group_ids) for m, s in scores.items()}
    exact = {m: r["exact"] for m, r in ranks.items()}
    redundancy = {m: r["redundancy"] for m, r in ranks.items()}

    sections: list[str] = [f"# Error analysis - {args.dump.name}",
                           f"queries: {len(captions)} (V2T direction)\n"]
    summary: dict = {}

    # 1. caption length (words)
    word_counts = [len(c.split()) for c in captions]
    length_labels = [bin_label(w, args.length_bins) for w in word_counts]
    length_order = sorted(set(length_labels), key=lambda lab: (lab.startswith(">"), lab))
    rows = stratify(exact, length_labels, length_order)
    sections += ["## By caption length (words), exact scoring", markdown_table(rows)]
    summary["by_caption_length"] = rows

    # 2. caption-group size (unique vs duplicated captions)
    group_sizes = torch.bincount(group_ids)[group_ids]
    size_labels = ["1 (unique)" if s == 1 else ("2-4" if s <= 4 else ">=5") for s in group_sizes.tolist()]
    size_order = ["1 (unique)", "2-4", ">=5"]
    rows_exact = stratify(exact, size_labels, size_order)
    rows_red = stratify(redundancy, size_labels, size_order)
    sections += ["## By caption-group size - exact scoring", markdown_table(rows_exact),
                 "## By caption-group size - redundancy-aware scoring", markdown_table(rows_red)]
    summary["by_group_size_exact"] = rows_exact
    summary["by_group_size_redundancy"] = rows_red

    # 3. clip length in frames (capped at max-frames during encoding)
    frame_labels = [bin_label(int(f), args.frame_bins) for f in lengths.tolist()]
    frame_order = sorted(set(frame_labels), key=lambda lab: (lab.startswith(">"), int(lab.split("-")[0].lstrip(">"))))
    rows = stratify(exact, frame_labels, frame_order)
    sections += ["## By clip length (frames, capped)", markdown_table(rows)]
    summary["by_frames"] = rows

    # 3b. optional manifest-joined strata (category, hand-detection ratio, ...)
    if args.manifest:
        uids: list[str] = dump.get("uids", [])
        for column in args.join_columns:
            labels = manifest_strata_labels(args.manifest, uids, column)
            if labels is None:
                continue
            order = sorted(set(labels))
            rows = stratify(exact, labels, order)
            sections += [f"## By manifest column `{column}`", markdown_table(rows)]
            summary[f"by_{column}"] = rows

    # 4. hubness before/after Sinkhorn
    sections.append("## Hub candidates in top-10 lists (V2T)")
    summary["hubs"] = {}
    for method in ("cosine", "sinkhorn"):
        stats = hub_stats(scores[method], captions, args.hub_top)
        summary["hubs"][method] = stats
        sections.append(
            f"**{method}**: {stats['distinct_candidates_in_top10']} distinct candidates fill the "
            f"top-10 slots; the 100 most frequent take "
            f"{100 * stats['share_of_slots_taken_by_top100_hubs']:.1f}% of all slots.\n"
        )
        sections.append(markdown_table(stats["top_hubs"]))

    # 5. queries fixed / broken by Sinkhorn (top-10 membership)
    in10_cos, in10_sink = exact["cosine"] <= 10, exact["sinkhorn"] <= 10
    fixed = int((~in10_cos & in10_sink).sum())
    broken = int((in10_cos & ~in10_sink).sum())
    sections.append(f"## Sinkhorn vs cosine transitions\nfixed (entered top-10): {fixed}  |  "
                    f"broken (left top-10): {broken}  |  net {fixed - broken:+d}\n")
    summary["sinkhorn_transitions"] = {"fixed": fixed, "broken": broken}

    # 6. worst V2T failures under Sinkhorn, with retrieved neighbours
    worst = torch.argsort(exact["sinkhorn"], descending=True)[: args.top_failures]
    top_idx = torch.topk(scores["sinkhorn"], k=args.neighbours, dim=1).indices
    failure_rows = []
    for q in worst.tolist():
        failure_rows.append({
            "query_caption": captions[q][:70],
            "words": word_counts[q],
            "frames": int(lengths[q]),
            "group_size": int(group_sizes[q]),
            "rank_cosine": int(exact["cosine"][q]),
            "rank_sinkhorn": int(exact["sinkhorn"][q]),
            "retrieved": " || ".join(captions[i][:45] for i in top_idx[q].tolist()),
        })
    sections += [f"## Worst {args.top_failures} V2T failures (Sinkhorn, exact)", markdown_table(failure_rows)]
    summary["worst_failures"] = failure_rows

    report_text = "\n".join(sections)
    print(report_text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")
        print(f"\nreport saved: {args.report}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary saved: {args.out}")


if __name__ == "__main__":
    main()
