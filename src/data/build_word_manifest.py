"""Build a word-level pretraining manifest from the vsl-recognition sequences.

The recognition project stores isolated-sign keypoint sequences as::

    <sequences_root>/<Word>/<video>.npy         # original
    <sequences_root>/<Word>/<video>_aug<N>.npy   # augmented

Each ``.npy`` is ``(frames, 1662)`` and is already shoulder-normalized in exactly
the same way as the sentence pipeline (``src/keypoints/holistic.py``), so the
clips drop straight into :class:`RetrievalDataset` for Stage-A sign pretraining.

Two CSV manifests are emitted with the columns ``RetrievalDataset`` needs
(``keypoint_path``, ``canonical_text``). Samples of the same word share a
caption, so the trainer's redundancy grouping turns them into mutual positives -
i.e. Stage-A becomes word<->text contrastive with no extra training code.

The split is leakage-free, mirroring the recognition data loader: a fraction of
the ORIGINAL clips per word is held out for validation, while every augmented
clip stays in train (and at least one original always remains in train).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


AUG_MARKER = "_aug"


def clean_word(folder_name: str) -> str:
    """Folder name -> caption text, e.g. ``Good_Morning`` -> ``good morning``."""
    text = folder_name.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


def is_augmented(npy_path: Path) -> bool:
    return AUG_MARKER in npy_path.stem


def build_rows(sequences_root: Path, val_fraction: float, seed: int):
    rng = random.Random(seed)
    word_dirs = sorted(d for d in sequences_root.iterdir() if d.is_dir())
    if not word_dirs:
        raise SystemExit(f"No word folders under {sequences_root}")

    train_rows: list[tuple[str, str, str, int]] = []
    val_rows: list[tuple[str, str, str, int]] = []
    for word_dir in word_dirs:
        caption = clean_word(word_dir.name)
        npy_files = sorted(word_dir.glob("*.npy"))
        originals = [f for f in npy_files if not is_augmented(f)]
        augmented = [f for f in npy_files if is_augmented(f)]
        if not originals:
            continue

        # Hold out a fraction of ORIGINALS for val; keep >=1 original in train.
        rng.shuffle(originals)
        n_val = min(int(len(originals) * val_fraction), len(originals) - 1)
        val_originals = originals[:n_val]
        train_originals = originals[n_val:]

        for f in train_originals:
            train_rows.append((str(f), caption, word_dir.name, 1))
        for f in augmented:
            train_rows.append((str(f), caption, word_dir.name, 0))
        for f in val_originals:
            val_rows.append((str(f), caption, word_dir.name, 1))

    return train_rows, val_rows


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keypoint_path", "canonical_text", "word", "is_original"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences-root",
        type=Path,
        default=Path("/mnt/ngan/recognition/sequences"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows, val_rows = build_rows(args.sequences_root, args.val_fraction, args.seed)
    train_path = args.out_dir / "word_pretrain_train.csv"
    val_path = args.out_dir / "word_pretrain_val.csv"
    write_csv(train_path, train_rows)
    write_csv(val_path, val_rows)

    n_words = len({r[2] for r in train_rows})
    n_aug = sum(1 for r in train_rows if r[3] == 0)
    print(f"words          : {n_words}")
    print(f"train rows     : {len(train_rows)} ({len(train_rows) - n_aug} orig + {n_aug} aug)")
    print(f"val rows       : {len(val_rows)} (originals only)")
    print(f"train manifest : {train_path}")
    print(f"val manifest   : {val_path}")


if __name__ == "__main__":
    main()
