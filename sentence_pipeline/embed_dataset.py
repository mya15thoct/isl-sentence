"""
Extract bilstm2 embeddings for all 491 sentence sequences.

Output: ISL-Sequences/sentence_embeddings/
  embeddings.npz  — {'X': list of (T_i, 64), 'y': (491,), 'sentences': (491,)}
  class_mapping.json

Usage:
  python sentence_pipeline/embed_dataset.py
"""

import json
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import ISL_SEQ_DIR
from sentence_pipeline.word_encoder import WordEncoder


WORD_MODEL  = '/mnt/ngan/recognition/checkpoints/mlp/best_model'
SEQ_DIR     = ISL_SEQ_DIR / 'keypoints'
OUT_DIR     = ISL_SEQ_DIR / 'sentence_embeddings'


def load_sequences(seq_dir: Path):
    class_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
    class2id   = {d.name: i for i, d in enumerate(class_dirs)}
    id2class   = {i: d.name for i, d in enumerate(class_dirs)}

    sequences, labels, names = [], [], []
    for cd in class_dirs:
        cid = class2id[cd.name]
        for npy in sorted(cd.glob('*.npy')):
            seq = np.load(npy).astype(np.float32)
            sequences.append(seq)
            labels.append(cid)
            names.append(cd.name)

    print(f'Loaded {len(sequences)} sequences  |  {len(class2id)} classes')
    return sequences, np.array(labels), names, class2id, id2class


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sequences, labels, names, class2id, id2class = load_sequences(SEQ_DIR)

    print('Extracting bilstm2 embeddings ...')
    encoder = WordEncoder(WORD_MODEL)
    embeddings = encoder.encode_batch(sequences, batch_size=64)

    shapes = [e.shape for e in embeddings]
    print(f'  Embedding shapes (first 5): {shapes[:5]}')
    print(f'  Min T: {min(s[0] for s in shapes)}  Max T: {max(s[0] for s in shapes)}')

    # Save — embeddings are variable-length so use object array
    emb_array = np.empty(len(embeddings), dtype=object)
    for i, e in enumerate(embeddings):
        emb_array[i] = e

    out_path = OUT_DIR / 'embeddings.npz'
    np.savez(out_path,
             embeddings = emb_array,
             labels     = labels,
             names      = np.array(names))
    print(f'Saved → {out_path}')

    with open(OUT_DIR / 'class_mapping.json', 'w') as f:
        json.dump({'class2id': class2id,
                   'id2class': {str(k): v for k, v in id2class.items()}}, f, indent=2)
    print(f'Saved → {OUT_DIR / "class_mapping.json"}')


if __name__ == '__main__':
    main()
