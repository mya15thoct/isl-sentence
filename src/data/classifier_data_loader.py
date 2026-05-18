"""
Data Loader for Sentence Classifier (Approach D).

Key differences from data_loader.py (CTC):
  - Label = sentence_id (single int, 0-indexed)  instead of [gloss, gloss, ...]
  - No CTC constraint (T >= label_len)
  - No blank token
  - Stratified train/val split: each sentence class gets at least 1 val sample
  - Returns (X, y) pairs where y is (B,) int array

Usage:
  train_gen, val_gen, meta = build_classifier_datasets(
      seq_dir    = '/mnt/ngan/ISL-Sequences/sequences',
      label_path = '/mnt/ngan/ISL-Sequences/labels.json',
  )
  # meta['num_sentences']   → 99
  # meta['sentences_sorted'] → ['he is crying', 'i am happy', ...]
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR

# Reuse helpers from data_loader.py to avoid duplication
from data.data_loader import augment_keypoints, _find_folder


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier_samples(
    seq_dir:    str,
    label_path: str,
) -> tuple[list[dict], dict, list[str]]:
    """
    Scan sequences/ folder and assign each .npy file a sentence class ID.

    Sentence → ID mapping is alphabetically stable so inference is reproducible.

    Returns:
        samples:          list of dicts {path, sentence_id, T, sentence, glosses}
        sentence2id:      {'he is crying': 0, 'i am happy': 1, ...}
        sentences_sorted: list of sentence strings in ID order
    """
    seq_dir = Path(seq_dir)

    with open(label_path, encoding='utf-8') as f:
        labels = json.load(f)   # {sentence_str: [gloss, ...]}

    sentences_sorted = sorted(labels.keys())
    sentence2id = {s: i for i, s in enumerate(sentences_sorted)}

    samples = []
    missing = []

    for sentence in sentences_sorted:
        folder = _find_folder(seq_dir, sentence)
        if folder is None:
            missing.append(sentence)
            continue

        sid = sentence2id[sentence]
        for npy_file in sorted(folder.glob('*.npy')):
            arr = np.load(npy_file, mmap_mode='r')
            samples.append({
                'path':        npy_file,
                'sentence_id': sid,
                'T':           arr.shape[0],
                'sentence':    sentence,
                'glosses':     labels[sentence],
            })

    print(f'[ClassifierLoader] {len(samples)} samples across '
          f'{len(sentences_sorted)} sentence classes')
    if missing:
        print(f'  Missing sequence folders ({len(missing)}): '
              f'{missing[:5]}{"..." if len(missing) > 5 else ""}')

    return samples, sentence2id, sentences_sorted


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator for sentence classification training.

    Each batch returns:
      X : (B, T_max, 1662)   float32 — padded keypoint sequences
      y : (B,)               int32   — sentence class IDs

    augment_factor > 1 tiles the sample list so each sample appears N times
    per epoch, each time with a different random augmentation.
    """

    def __init__(
        self,
        samples:        list[dict],
        batch_size:     int  = 8,
        shuffle:        bool = True,
        augment:        bool = False,
        augment_factor: int  = 1,
    ):
        self.samples    = samples
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.augment    = augment

        base = np.arange(len(samples))
        self.indices = np.tile(base, max(1, augment_factor))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch     = [self.samples[i] for i in batch_idx]

        rng    = np.random.default_rng() if self.augment else None
        arrays = []
        for s in batch:
            arr = np.load(s['path'])
            if self.augment:
                arr = augment_keypoints(arr, rng)
            arrays.append(arr)

        max_T = max(a.shape[0] for a in arrays)
        B     = len(batch)

        X = np.zeros((B, max_T, 1662), dtype=np.float32)
        y = np.array([batch[i]['sentence_id'] for i in range(B)], dtype=np.int32)

        for i, arr in enumerate(arrays):
            X[i, :arr.shape[0], :] = arr

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier_datasets(
    seq_dir:        str   = None,
    label_path:     str   = None,
    batch_size:     int   = 8,
    val_split:      float = 0.15,
    seed:           int   = 42,
    augment_factor: int   = 10,
) -> tuple:
    """
    Build stratified train/val generators for sentence classification.

    Stratified split ensures every sentence class has at least 1 val sample.
    With ~5 videos/sentence and val_split=0.15:
      - 1 val sample per class  (→ 99 total val samples)
      - 4 train samples per class (→ 396 total, ×10 aug = 3960 effective)

    Returns:
        train_gen : ClassifierDataGenerator
        val_gen   : ClassifierDataGenerator
        meta      : dict with num_sentences, sentence2id, sentences_sorted, etc.
    """
    seq_dir    = seq_dir    or str(ISL_SEQ_DIR / 'keypoints')
    label_path = label_path or str(ISL_SEQ_DIR / 'labels.json')

    samples, sentence2id, sentences_sorted = load_classifier_samples(
        seq_dir, label_path)

    rng = np.random.default_rng(seed)

    # Stratified split by sentence class
    by_class = defaultdict(list)
    for s in samples:
        by_class[s['sentence_id']].append(s)

    train_s, val_s = [], []
    for sid in sorted(by_class.keys()):
        class_samples = by_class[sid].copy()
        rng.shuffle(class_samples)
        n_val = max(1, int(len(class_samples) * val_split))
        val_s.extend(class_samples[:n_val])
        train_s.extend(class_samples[n_val:])

    aug = max(1, augment_factor)
    print(f'[ClassifierLoader] Train: {len(train_s)} '
          f'(×{aug} aug = {len(train_s) * aug} effective)  |  '
          f'Val: {len(val_s)}')

    train_gen = ClassifierDataGenerator(
        train_s, batch_size=batch_size, shuffle=True,
        augment=True, augment_factor=aug,
    )
    val_gen = ClassifierDataGenerator(
        val_s, batch_size=batch_size, shuffle=False,
        augment=False, augment_factor=1,
    )

    meta = {
        'num_sentences':   len(sentences_sorted),
        'sentence2id':     sentence2id,
        'id2sentence':     {v: k for k, v in sentence2id.items()},
        'sentences_sorted': sentences_sorted,
        'num_train':       len(train_s),
        'num_val':         len(val_s),
        'augment_factor':  aug,
    }

    return train_gen, val_gen, meta


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_gen, val_gen, meta = build_classifier_datasets()

    print(f'\nSentence classes : {meta["num_sentences"]}')
    print(f'Train batches    : {len(train_gen)}')
    print(f'Val   batches    : {len(val_gen)}')

    X, y = train_gen[0]
    print(f'\nFirst batch:')
    print(f'  X shape : {X.shape}')
    print(f'  y       : {y}')
    print(f'  Sentences: {[meta["id2sentence"][i] for i in y[:4]]}')
