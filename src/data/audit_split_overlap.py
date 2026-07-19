"""Cross-split overlap audit for arbitrary grouping keys (review Issue 6).

The video-grouped split guarantees zero video_id overlap, but the reviewer asks
whether the SAME signer / session / source channel spans splits through
*different* video ids. iSign carries no explicit signer metadata, so this tool
audits whatever proxy keys the manifests do expose:

  * ``--key-column video_id``                    -> re-verifies the zero-leakage claim;
  * ``--key-column video_id --key-regex REGEX``  -> audits a proxy extracted from the
    id (e.g. a channel/session prefix), where REGEX has one capture group;
  * any other manifest column via --key-column (use --list-columns to inspect).

Reports per-split unique keys, every pairwise intersection (count + share +
examples), and exits non-zero with --expect-disjoint if any overlap is found.
Whatever the outcome, the paper should disclose which keys were auditable and
that unseen-signer generalization remains an open limitation if signer ids are
unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.dataset import read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", help="one per manifest (default: file stems)")
    parser.add_argument("--key-column", default="video_id")
    parser.add_argument("--key-regex", help="regex with one capture group applied to the key column")
    parser.add_argument("--list-columns", action="store_true", help="print available columns and exit")
    parser.add_argument("--expect-disjoint", action="store_true", help="exit 1 if any overlap is found")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--out", type=Path, help="optional JSON output")
    return parser.parse_args()


def extract_keys(rows: list[dict], column: str, pattern: re.Pattern | None) -> tuple[set[str], int]:
    keys: set[str] = set()
    unmatched = 0
    for row in rows:
        value = row.get(column, "").strip()
        if not value:
            unmatched += 1
            continue
        if pattern is not None:
            match = pattern.search(value)
            if match is None:
                unmatched += 1
                continue
            value = match.group(1)
        keys.add(value)
    return keys, unmatched


def main() -> None:
    args = parse_args()
    labels = args.labels or [m.stem for m in args.manifests]
    if len(labels) != len(args.manifests):
        raise SystemExit("--labels must match --manifests")

    tables = {label: read_csv(manifest) for label, manifest in zip(labels, args.manifests)}

    if args.list_columns:
        for label, rows in tables.items():
            print(f"{label}: {sorted(rows[0].keys()) if rows else '(empty)'}")
        return

    pattern = re.compile(args.key_regex) if args.key_regex else None
    key_desc = args.key_column + (f" ~ /{args.key_regex}/" if args.key_regex else "")
    print(f"key: {key_desc}\n")

    keys: dict[str, set[str]] = {}
    for label, rows in tables.items():
        if rows and args.key_column not in rows[0]:
            raise SystemExit(
                f"column {args.key_column!r} not in {label}; available: {sorted(rows[0].keys())}"
            )
        split_keys, unmatched = extract_keys(rows, args.key_column, pattern)
        keys[label] = split_keys
        note = f"  ({unmatched} rows without a key!)" if unmatched else ""
        print(f"{label:<8} rows={len(rows):<8} unique keys={len(split_keys)}{note}")

    report: dict[str, dict] = {"key": key_desc, "pairs": {}}
    any_overlap = False
    print()
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = keys[a] & keys[b]
            any_overlap = any_overlap or bool(shared)
            share = len(shared) / max(1, min(len(keys[a]), len(keys[b])))
            examples = sorted(shared)[: args.examples]
            report["pairs"][f"{a}&{b}"] = {"shared": len(shared), "share_of_smaller": round(share, 4),
                                           "examples": examples}
            status = "OK (disjoint)" if not shared else f"OVERLAP ({100 * share:.1f}% of smaller split)"
            print(f"{a} & {b}: {len(shared)} shared keys -> {status}")
            if examples and shared:
                print(f"    e.g. {examples}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nsaved: {args.out}")

    if args.expect_disjoint and any_overlap:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
