"""
Build Attention Dictionary [D] from trained Word Model.

D[gloss] = attention-weighted average gate weight vector (3,)
         = (pose_importance, face_importance, hand_importance)

Used by Sentence Model Distillation Loss:
  L_distill = MSE(gate_weight_frame_t, D[predicted_gloss_t])

Usage:
  python src/utils/build_dictionary.py
  python src/utils/build_dictionary.py --model_path path/to/best_model
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))   # vsl-sentence root
import tensorflow as tf

from config import (
    WORD_MODEL_PATH, WORD_SEQUENCE_PATH, DICTIONARY_SAVE_PATH
)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_extractor(word_model: tf.keras.Model) -> tf.keras.Model:
    """Extract pose/face/hand features + temporal attention from Word Model."""
    outputs = {
        'pose': word_model.get_layer('pose_features').output,      # (B,T,64)
        'face': word_model.get_layer('face_features').output,      # (B,T,128)
        'hand': word_model.get_layer('hand_features').output,      # (B,T,64)
        'attn': word_model.get_layer('temporal_attention').output,  # (B,T,1)
    }
    return tf.keras.Model(inputs=word_model.input, outputs=outputs,
                          name='feature_extractor')


# ─────────────────────────────────────────────────────────────────────────────
# GATE WEIGHT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_gate_weight(pose_feat, face_feat, hand_feat, attn_weights):
    """
    Compute attention-weighted gate weight for one sequence.

    g_t = softmax([||pose_t||₂, ||face_t||₂, ||hand_t||₂])   per frame
    G   = Σ_t (α_t × g_t)   weighted by temporal attention

    Returns: (3,) — [pose_importance, face_importance, hand_importance]
    """
    norms     = np.stack([
        np.linalg.norm(pose_feat, axis=-1),
        np.linalg.norm(face_feat, axis=-1),
        np.linalg.norm(hand_feat, axis=-1),
    ], axis=-1)                                                     # (T, 3)
    exp_norms = np.exp(norms - norms.max(axis=-1, keepdims=True))
    g_t       = exp_norms / exp_norms.sum(axis=-1, keepdims=True)  # (T, 3)

    alpha = attn_weights[:, 0]
    alpha = alpha / (alpha.sum() + 1e-8)

    return (alpha[:, np.newaxis] * g_t).sum(axis=0).astype(np.float32)  # (3,)


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

def build_attention_dictionary(
    model_path:    str  = None,
    sequence_path: Path = None,
    save_path:     str  = None,
) -> dict:
    """
    Build D from all word sequences in sequence_path.

    Saves:
      <save_path>/dictionary.npy    ← used in training
      <save_path>/dictionary.json   ← human-readable
      <save_path>/action_names.json ← gloss index mapping

    Returns: {gloss_name: np.array(3,)}
    """
    model_path    = model_path    or str(WORD_MODEL_PATH)
    sequence_path = Path(sequence_path or WORD_SEQUENCE_PATH)
    save_path     = save_path     or str(DICTIONARY_SAVE_PATH)

    print("=" * 60)
    print("BUILDING ATTENTION DICTIONARY [D]")
    print("=" * 60)

    print(f"\nWord Model  : {model_path}")
    print(f"Sequences   : {sequence_path}")
    print(f"Save to     : {save_path}")

    word_model        = tf.keras.models.load_model(model_path)
    feature_extractor = build_feature_extractor(word_model)

    # Lấy sequence length từ model thay vì hardcode
    seq_len = word_model.input_shape[1]
    print(f"Model sequence length: {seq_len}")
    print("\nFeature extractor built.\n")

    gloss_folders = sorted([d for d in sequence_path.iterdir() if d.is_dir()])
    print(f"Found {len(gloss_folders)} gloss classes.\n")

    D            = {}
    action_names = []

    for idx, gloss_folder in enumerate(gloss_folders):
        gloss_name = gloss_folder.name
        action_names.append(gloss_name)

        npy_files = sorted(gloss_folder.glob('*.npy'))
        if not npy_files:
            print(f"  [SKIP] {gloss_name} — no .npy files found")
            continue

        gate_weights = []
        for npy_file in npy_files:
            seq      = np.load(npy_file).astype(np.float32)
            T        = seq.shape[0]
            padded   = np.zeros((seq_len, 1662), dtype=np.float32)
            real_len = min(T, seq_len)
            padded[:real_len] = seq[:real_len]

            feats  = feature_extractor(padded[np.newaxis], training=False)
            pose_f = feats['pose'].numpy()[0][:real_len]
            face_f = feats['face'].numpy()[0][:real_len]
            hand_f = feats['hand'].numpy()[0][:real_len]
            alpha  = feats['attn'].numpy()[0][:real_len]

            gate_weights.append(compute_gate_weight(pose_f, face_f, hand_f, alpha))

        D[gloss_name] = np.mean(gate_weights, axis=0)
        print(f"  [{idx+1:3d}/{len(gloss_folders)}] {gloss_name:30s} "
              f"pose={D[gloss_name][0]:.3f}  "
              f"face={D[gloss_name][1]:.3f}  "
              f"hand={D[gloss_name][2]:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(f"{save_path}/dictionary.npy", D)

    with open(f"{save_path}/dictionary.json", 'w') as f:
        json.dump({k: v.tolist() for k, v in D.items()}, f, indent=2)

    with open(f"{save_path}/action_names.json", 'w') as f:
        json.dump(action_names, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[OK] DICTIONARY BUILT — {len(D)} glosses")
    print(f"  {save_path}/dictionary.npy    ← training")
    print(f"  {save_path}/dictionary.json   ← inspect")
    print(f"  {save_path}/action_names.json ← index map")
    print(f"{'=' * 60}")

    return D


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',    type=str, default=None)
    parser.add_argument('--sequence_path', type=str, default=None)
    parser.add_argument('--save_path',     type=str, default=None)
    args = parser.parse_args()

    build_attention_dictionary(
        model_path=args.model_path,
        sequence_path=args.sequence_path,
        save_path=args.save_path,
    )
