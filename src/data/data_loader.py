"""
Data Loader for CTC Sentence Training.

Loads extracted .npy keypoint sequences and labels.json,
maps gloss names → indices, pads variable-length sequences,
and returns batches compatible with CTC loss.

Usage:
  train_ds, val_ds, meta = build_datasets(
      seq_dir   = '/mnt/ngan/ISL-Sequences/sequences',
      label_path= '/mnt/ngan/ISL-Sequences/labels.json',
      vocab_path = '/mnt/ngan/ISL-Sequences/checkpoints/action_mapping_combined.json',
  )
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR, ACTION_MAPPING_PATH


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sentence_to_folder(sentence: str) -> str:
    """'i am crying' → 'i_am_crying'  (matches extract_sentences.py naming)"""
    return sentence.strip().lower().replace(' ', '_')


def _find_folder(seq_dir: Path, sentence: str) -> Path | None:
    """
    Find the sequence folder for a sentence, case-insensitive.
    Handles variations like:
      - 'He_is_going...' vs 'he_is_going...'
      - 'which_collegeschool...' vs 'which_college_school...'
    """
    target = _sentence_to_folder(sentence)

    # Exact match first
    if (seq_dir / target).exists():
        return seq_dir / target

    # Case-insensitive + normalize (remove all underscores for comparison)
    target_norm = target.replace('_', '')
    for folder in seq_dir.iterdir():
        if not folder.is_dir():
            continue
        folder_norm = folder.name.lower().replace('_', '')
        if folder_norm == target_norm:
            return folder

    return None


def load_vocab(vocab_path: str) -> dict:
    """
    Load action_mapping_combined.json and build gloss→index map.
    CTC blank token is assigned index 0; all gloss indices are shifted +1.

    Returns:
        gloss2idx: {'WATER': 1, 'HELP': 2, ...}   (1-indexed, 0=blank)
    """
    with open(vocab_path, encoding='utf-8') as f:
        mapping = json.load(f)   # {str_idx: gloss_name}

    # Sort by original index to preserve ordering
    sorted_glosses = [v for _, v in sorted(mapping.items(), key=lambda x: int(x[0]))]

    # 0-indexed: glosses = 0..N-1, blank is at index N (handled by ctc_batch_cost)
    gloss2idx = {g.upper(): i for i, g in enumerate(sorted_glosses)}
    return gloss2idx


def load_active_vocab(vocab_path: str, label_path: str) -> dict:
    """
    Load vocab filtered to only glosses appearing in sentence labels.
    Re-indexes from 0..N-1, drastically reducing CTC search space.

    With 99 sentences using ~80 unique glosses, this shrinks the output
    from 410 (409+blank) to ~81 (80+blank), making CTC 5× easier.
    """
    full_gloss2idx = load_vocab(vocab_path)

    with open(label_path, encoding='utf-8') as f:
        labels = json.load(f)

    active = set()
    for glosses in labels.values():
        for g in glosses:
            if g.upper() in full_gloss2idx:
                active.add(g.upper())

    active_sorted = sorted(active)
    gloss2idx = {g: i for i, g in enumerate(active_sorted)}

    print(f'[DataLoader] Active vocab: {len(gloss2idx)} glosses '
          f'(reduced from {len(full_gloss2idx)})')
    return gloss2idx


def load_samples(
    seq_dir:    str,
    label_path: str,
    gloss2idx:  dict,
) -> list[dict]:
    """
    Scan sequences/ folder and pair each .npy file with its CTC label.

    Returns list of dicts:
      {
        'path':        Path to .npy file,
        'label':       list of int (gloss indices),
        'T':           int (number of frames),
        'label_len':   int,
        'sentence':    str,
      }
    """
    seq_dir = Path(seq_dir)

    with open(label_path, encoding='utf-8') as f:
        labels = json.load(f)   # {sentence: [gloss, gloss, ...]}

    samples = []
    missing_folders = []
    ctc_violations  = []

    for sentence, glosses in labels.items():
        folder_path = _find_folder(seq_dir, sentence)

        if folder_path is None:
            missing_folders.append(sentence)
            continue

        # Map gloss names → indices (uppercase, skip unknown)
        label_indices = [gloss2idx[g.upper()] for g in glosses
                         if g.upper() in gloss2idx]

        if not label_indices:
            continue

        label_len = len(label_indices)

        for npy_file in sorted(folder_path.glob('*.npy')):
            # Peek at shape without loading full array
            arr  = np.load(npy_file, mmap_mode='r')
            T    = arr.shape[0]

            # CTC constraint: input_length >= label_length
            if T < label_len:
                ctc_violations.append(f'{npy_file.name} (T={T} < L={label_len})')
                continue

            samples.append({
                'path':      npy_file,
                'label':     label_indices,
                'T':         T,
                'label_len': label_len,
                'sentence':  sentence,
            })

    print(f'[DataLoader] Loaded {len(samples)} samples')
    if missing_folders:
        print(f'  Missing folders ({len(missing_folders)}): {missing_folders[:5]}...')
    if ctc_violations:
        print(f'  CTC violations skipped ({len(ctc_violations)}): {ctc_violations[:3]}')

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION (keypoint-level)
# ─────────────────────────────────────────────────────────────────────────────

def augment_keypoints(
    seq: np.ndarray,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply random augmentation to a keypoint sequence (T, 1662).

    Augmentations (each applied with probability ~50-70%):
      1. Gaussian noise          — simulate sensor jitter
      2. Random scale            — simulate distance variation
      3. Speed perturbation      — simulate signing speed variation
      4. Frame dropout           — simulate occlusion
      5. Random coordinate shift — simulate camera shift
    """
    if rng is None:
        rng = np.random.default_rng()

    aug = seq.copy()

    # 1. Gaussian noise (σ=0.005)
    if rng.random() > 0.3:
        aug += rng.normal(0, 0.005, aug.shape).astype(np.float32)

    # 2. Random scale (0.9–1.1)
    if rng.random() > 0.5:
        scale = rng.uniform(0.9, 1.1)
        aug *= scale

    # 3. Speed perturbation (0.8–1.2× temporal stretch)
    if rng.random() > 0.3:
        factor = rng.uniform(0.8, 1.2)
        T = aug.shape[0]
        new_T = max(1, int(T * factor))
        indices = np.linspace(0, T - 1, new_T).astype(int)
        aug = aug[indices]

    # 4. Frame dropout (zero out 5–15% of frames)
    if rng.random() > 0.5:
        T = aug.shape[0]
        n_drop = rng.integers(1, max(2, int(T * 0.15)))
        drop_idx = rng.choice(T, n_drop, replace=False)
        aug[drop_idx] = 0.0

    # 5. Random coordinate shift
    if rng.random() > 0.5:
        shift = rng.uniform(-0.02, 0.02, (1, 1662)).astype(np.float32)
        aug += shift

    return aug.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# TENSORFLOW DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _pad_sequences(batch: list[dict], augment: bool = False) -> tuple:
    """
    Pad a batch of variable-length sequences to the same T.
    Optionally applies keypoint augmentation (for training).
    Returns tensors compatible with CTC loss.
    """
    rng = np.random.default_rng() if augment else None

    # Load arrays (and optionally augment — may change T)
    arrays = []
    for s in batch:
        arr = np.load(s['path'])
        if augment:
            arr = augment_keypoints(arr, rng)
        arrays.append(arr)

    max_T  = max(a.shape[0] for a in arrays)
    max_L  = max(s['label_len'] for s in batch)
    B      = len(batch)

    X          = np.zeros((B, max_T, 1662), dtype=np.float32)
    y          = np.zeros((B, max_L),       dtype=np.int32)
    in_lens    = np.zeros((B,),             dtype=np.int32)
    label_lens = np.zeros((B,),             dtype=np.int32)

    for i, (s, arr) in enumerate(zip(batch, arrays)):
        T = arr.shape[0]
        # CTC constraint: input_length >= label_length
        if T < s['label_len']:
            indices = np.linspace(0, T - 1, s['label_len']).astype(int)
            arr = arr[indices]
            T = arr.shape[0]
        X[i, :T, :]           = arr
        y[i, :s['label_len']] = s['label']
        in_lens[i]            = T
        label_lens[i]         = s['label_len']

    return X, y, in_lens, label_lens


class SentenceDataGenerator(tf.keras.utils.Sequence):
    """
    Keras-compatible data generator for CTC sentence training.

    Each batch returns:
      inputs  : (B, T_max, 1662)   padded sequences
      labels  : (B, L_max)         padded label indices
      in_lens : (B,)               actual frame counts
      lbl_lens: (B,)               actual label lengths

    When augment=True, keypoint-level augmentation is applied on-the-fly.
    augment_factor > 1 repeats each sample N times per epoch (different
    random augmentation each time), effectively multiplying dataset size.
    """

    def __init__(
        self,
        samples:         list[dict],
        batch_size:      int  = 8,
        shuffle:         bool = True,
        augment:         bool = False,
        augment_factor:  int  = 1,
    ):
        self.samples    = samples
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.augment    = augment
        # Tile indices to see each sample augment_factor times per epoch
        base = np.arange(len(samples))
        self.indices = np.tile(base, max(1, augment_factor))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch     = [self.samples[i] for i in batch_idx]
        X, y, in_lens, lbl_lens = _pad_sequences(batch, augment=self.augment)
        return {
            'inputs':         X,
            'labels':         y,
            'input_lengths':  in_lens,
            'label_lengths':  lbl_lens,
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    seq_dir:        str   = None,
    label_path:     str   = None,
    vocab_path:     str   = None,
    batch_size:     int   = 8,
    val_split:      float = 0.15,
    seed:           int   = 42,
    active_only:    bool  = False,
    augment_factor: int   = 0,
):
    """
    Build train/val data generators and return metadata.

    Args:
        active_only:    If True, reduce vocab to only glosses in labels (~80 vs 409)
        augment_factor: Multiply training data N× with random augmentation (0=off)

    Returns:
        train_gen : SentenceDataGenerator
        val_gen   : SentenceDataGenerator
        meta      : dict with vocab info, num_samples, etc.
    """
    seq_dir    = seq_dir    or str(ISL_SEQ_DIR / 'keypoints')
    label_path = label_path or str(ISL_SEQ_DIR / 'labels.json')
    vocab_path = vocab_path or str(ACTION_MAPPING_PATH)

    # Vocab: full (409) or active-only (~80)
    if active_only:
        gloss2idx = load_active_vocab(vocab_path, label_path)
    else:
        gloss2idx = load_vocab(vocab_path)

    samples = load_samples(seq_dir, label_path, gloss2idx)

    # Reproducible split
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)

    n_val   = max(1, int(len(samples) * val_split))
    val_s   = samples[:n_val]
    train_s = samples[n_val:]

    aug = max(1, augment_factor) if augment_factor > 0 else 1
    effective = len(train_s) * aug
    print(f'[DataLoader] Train: {len(train_s)} (×{aug}={effective})  |  Val: {len(val_s)}')
    print(f'[DataLoader] Vocab: {len(gloss2idx)} glosses  |  '
          f'CTC output: {len(gloss2idx) + 1} (+ blank)')

    train_gen = SentenceDataGenerator(
        train_s, batch_size=batch_size, shuffle=True,
        augment=(augment_factor > 0), augment_factor=aug,
    )
    val_gen = SentenceDataGenerator(
        val_s, batch_size=batch_size, shuffle=False,
        augment=False, augment_factor=1,
    )

    meta = {
        'gloss2idx':      gloss2idx,
        'idx2gloss':      {v: k for k, v in gloss2idx.items()},
        'num_glosses':    len(gloss2idx),
        'num_train':      len(train_s),
        'num_val':        len(val_s),
        'batch_size':     batch_size,
        'augment_factor': aug,
        'active_only':    active_only,
    }

    return train_gen, val_gen, meta


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_gen, val_gen, meta = build_datasets()

    print(f'\nVocab size : {meta["num_glosses"]} glosses (+ 1 blank = {meta["num_glosses"]+1})')
    print(f'Train batches: {len(train_gen)}  |  Val batches: {len(val_gen)}')

    # Inspect first batch
    batch = train_gen[0]
    print(f'\nBatch shapes:')
    print(f'  inputs        : {batch["inputs"].shape}')
    print(f'  labels        : {batch["labels"].shape}')
    print(f'  input_lengths : {batch["input_lengths"]}')
    print(f'  label_lengths : {batch["label_lengths"]}')
