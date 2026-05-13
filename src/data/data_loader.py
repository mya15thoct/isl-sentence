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

    # Blank = 0, glosses = 1..N
    gloss2idx = {g.upper(): i + 1 for i, g in enumerate(sorted_glosses)}
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
        folder_name = _sentence_to_folder(sentence)
        folder_path = seq_dir / folder_name

        if not folder_path.exists():
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
# TENSORFLOW DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _pad_sequences(batch: list[dict]) -> tuple:
    """
    Pad a batch of variable-length sequences to the same T.
    Returns tensors compatible with CTC loss.
    """
    max_T     = max(s['T'] for s in batch)
    max_L     = max(s['label_len'] for s in batch)
    B         = len(batch)

    X          = np.zeros((B, max_T, 1662), dtype=np.float32)
    y          = np.zeros((B, max_L),       dtype=np.int32)
    in_lens    = np.zeros((B,),             dtype=np.int32)
    label_lens = np.zeros((B,),             dtype=np.int32)

    for i, s in enumerate(batch):
        arr = np.load(s['path'])
        T   = s['T']
        X[i, :T, :]                    = arr
        y[i, :s['label_len']]          = s['label']
        in_lens[i]                     = T
        label_lens[i]                  = s['label_len']

    return X, y, in_lens, label_lens


class SentenceDataGenerator(tf.keras.utils.Sequence):
    """
    Keras-compatible data generator for CTC sentence training.

    Each batch returns:
      inputs  : (B, T_max, 1662)   padded sequences
      labels  : (B, L_max)         padded label indices
      in_lens : (B,)               actual frame counts
      lbl_lens: (B,)               actual label lengths
    """

    def __init__(
        self,
        samples:    list[dict],
        batch_size: int  = 8,
        shuffle:    bool = True,
    ):
        self.samples    = samples
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.indices    = np.arange(len(samples))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.samples) // self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch     = [self.samples[i] for i in batch_idx]
        X, y, in_lens, lbl_lens = _pad_sequences(batch)
        # Return as dict for custom CTC training step
        return {
            'inputs':       X,
            'labels':       y,
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
    seq_dir:    str  = None,
    label_path: str  = None,
    vocab_path: str  = None,
    batch_size: int  = 8,
    val_split:  float = 0.15,
    seed:       int  = 42,
):
    """
    Build train/val data generators and return metadata.

    Returns:
        train_gen : SentenceDataGenerator
        val_gen   : SentenceDataGenerator
        meta      : dict with vocab info, num_samples, etc.
    """
    seq_dir    = seq_dir    or str(ISL_SEQ_DIR / 'sequences')
    label_path = label_path or str(ISL_SEQ_DIR / 'labels.json')
    vocab_path = vocab_path or str(ACTION_MAPPING_PATH)

    gloss2idx = load_vocab(vocab_path)
    samples   = load_samples(seq_dir, label_path, gloss2idx)

    # Reproducible split
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)

    n_val   = max(1, int(len(samples) * val_split))
    val_s   = samples[:n_val]
    train_s = samples[n_val:]

    print(f'[DataLoader] Train: {len(train_s)}  |  Val: {len(val_s)}')

    train_gen = SentenceDataGenerator(train_s, batch_size=batch_size, shuffle=True)
    val_gen   = SentenceDataGenerator(val_s,   batch_size=batch_size, shuffle=False)

    meta = {
        'gloss2idx':   gloss2idx,
        'idx2gloss':   {v: k for k, v in gloss2idx.items()},
        'num_glosses': len(gloss2idx),   # excludes blank
        'num_train':   len(train_s),
        'num_val':     len(val_s),
        'batch_size':  batch_size,
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
