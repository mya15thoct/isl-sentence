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
    WORD_MODEL_PATH, WORD_SEQUENCE_PATH, ISL_WORD_PATH,
    ACTION_MAPPING_PATH, DICTIONARY_SAVE_PATH,
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
    # Normalize by sqrt(dims) to remove dimensionality bias before softmax
    # (face has 128 dims vs 64 for pose/hand — raw norm would always favor face)
    norms     = np.stack([
        np.linalg.norm(pose_feat, axis=-1) / np.sqrt(pose_feat.shape[-1]),
        np.linalg.norm(face_feat, axis=-1) / np.sqrt(face_feat.shape[-1]),
        np.linalg.norm(hand_feat, axis=-1) / np.sqrt(hand_feat.shape[-1]),
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
    model_path:          str  = None,
    video_sequence_path: str  = None,
    image_sequence_path: str  = None,
    action_mapping_path: str  = None,
    save_path:           str  = None,
) -> dict:
    """
    Build D from all word sequences, covering the full combined vocabulary.

    For each gloss in action_mapping_combined.json:
      - Prefers video sequences (recognition/sequences/)
      - Falls back to image sequences (ISL-Sequences/word/)

    Saves:
      <save_path>/dictionary.npy    ← used in training
      <save_path>/dictionary.json   ← human-readable
      <save_path>/action_names.json ← ordered gloss list (matches model output)

    Returns: {gloss_name: np.array(3,)}
    """
    model_path          = model_path          or str(WORD_MODEL_PATH)
    video_sequence_path = Path(video_sequence_path or WORD_SEQUENCE_PATH)
    image_sequence_path = Path(image_sequence_path or ISL_WORD_PATH)
    action_mapping_path = action_mapping_path  or str(ACTION_MAPPING_PATH)
    save_path           = save_path            or str(DICTIONARY_SAVE_PATH)

    print("=" * 60)
    print("BUILDING ATTENTION DICTIONARY [D]")
    print("=" * 60)
    print(f"\nWord Model     : {model_path}")
    print(f"Video seqs     : {video_sequence_path}")
    print(f"Image seqs     : {image_sequence_path}")
    print(f"Action mapping : {action_mapping_path}")
    print(f"Save to        : {save_path}")

    word_model        = tf.keras.models.load_model(model_path)
    feature_extractor = build_feature_extractor(word_model)
    seq_len           = word_model.input_shape[1]
    print(f"Model sequence length: {seq_len}")
    print("\nFeature extractor built.\n")

    # Use action_mapping to get full ordered class list (matches model output)
    with open(action_mapping_path) as f:
        mapping = json.load(f)
    action_names = [mapping[str(i)] for i in range(len(mapping))]
    print(f"Found {len(action_names)} gloss classes (from action_mapping).\n")

    D = {}

    for idx, gloss_name in enumerate(action_names):
        # Prefer video sequences; fall back to image sequences
        vid_folder = video_sequence_path / gloss_name
        img_folder = image_sequence_path / gloss_name

        if vid_folder.exists():
            npy_files = sorted(f for f in vid_folder.glob('*.npy')
                               if '_static' not in f.stem)
            source = 'video'
        else:
            npy_files = sorted(img_folder.glob('*.npy')) if img_folder.exists() else []
            source = 'image'

        if not npy_files:
            print(f"  [SKIP] {gloss_name} — no sequences found")
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
        print(f"  [{idx+1:3d}/{len(action_names)}] {gloss_name:30s} "
              f"pose={D[gloss_name][0]:.3f}  "
              f"face={D[gloss_name][1]:.3f}  "
              f"hand={D[gloss_name][2]:.3f}  [{source}]")

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
    parser.add_argument('--model_path',          type=str, default=None)
    parser.add_argument('--video_sequence_path', type=str, default=None)
    parser.add_argument('--image_sequence_path', type=str, default=None)
    parser.add_argument('--action_mapping_path', type=str, default=None)
    parser.add_argument('--save_path',           type=str, default=None)
    args = parser.parse_args()

    build_attention_dictionary(
        model_path=args.model_path,
        video_sequence_path=args.video_sequence_path,
        image_sequence_path=args.image_sequence_path,
        action_mapping_path=args.action_mapping_path,
        save_path=args.save_path,
    )
