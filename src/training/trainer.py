"""
Trainer for ISL Sentence Recognition (CTC + Distillation).

Training uses:
  1. CTC Loss       — aligns frame predictions with gloss sequences
  2. Distillation   — Attention Dictionary [D] guides body-part attention

Usage:
  python src/training/trainer.py
"""

import json
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR, ACTION_MAPPING_PATH, WORD_MODEL_PATH

sys.path.append(str(Path(__file__).parent.parent))
from data.data_loader import build_datasets
from models.sentence_model import build_sentence_model


# ─────────────────────────────────────────────────────────────────────────────
# CTC LOSS
# ─────────────────────────────────────────────────────────────────────────────

def ctc_loss_fn(labels, y_pred, input_lengths, label_lengths):
    """
    Compute CTC loss using tf.keras.backend.ctc_batch_cost.

    Args:
        labels:        (B, L_max)   padded gloss indices (0-indexed, blank at num_classes-1)
        y_pred:        (B, T, C)    softmax probabilities from model
        input_lengths: (B,)         actual frame counts
        label_lengths: (B,)         actual label lengths

    Returns:
        Scalar mean CTC loss
    """
    import tensorflow.keras.backend as K

    y_true      = tf.cast(labels,        tf.float32)
    y_pred_     = tf.cast(y_pred,        tf.float32)
    in_lens     = tf.cast(tf.reshape(input_lengths,  (-1, 1)), tf.int32)
    lbl_lens    = tf.cast(tf.reshape(label_lengths, (-1, 1)), tf.int32)

    loss = K.ctc_batch_cost(y_true, y_pred_, in_lens, lbl_lens)
    return tf.reduce_mean(loss)


# ─────────────────────────────────────────────────────────────────────────────
# DISTILLATION LOSS  (Attention Dictionary [D])
# ─────────────────────────────────────────────────────────────────────────────

def load_attention_dict(dict_dir: str) -> dict:
    """
    Load the attention dictionary built by build_dictionary.py.

    Returns:
        {GLOSS_NAME: np.array([pose_w, face_w, hand_w])}
    """
    dict_dir = Path(dict_dir)
    att_dict = {}

    # Try loading from JSON if available
    json_path = dict_dir / 'attention_weights.json'
    if json_path.exists():
        with open(json_path) as f:
            raw = json.load(f)
        for gloss, weights in raw.items():
            att_dict[gloss.upper()] = np.array(
                [weights['pose'], weights['face'], weights['hand']],
                dtype=np.float32,
            )
        return att_dict

    # Fallback: load per-gloss .npy files
    for npy_file in dict_dir.glob('*.npy'):
        gloss = npy_file.stem.upper()
        w = np.load(npy_file)   # shape (3,) or (sequence_length, 3)
        if w.ndim > 1:
            w = w.mean(axis=0)
        att_dict[gloss] = w.astype(np.float32)

    return att_dict


def distillation_loss_fn(
    logits:    tf.Tensor,        # (B, T, num_glosses+1)
    att_dict:  dict,             # gloss → [pose_w, face_w, hand_w]
    idx2gloss: dict,             # int → gloss name
    alpha:     float = 0.3,
) -> tf.Tensor:
    """
    Compute attention distillation loss.

    For each frame the model predicts a gloss probability distribution.
    The distillation loss penalises misalignment between the model's
    body-part attention and the expected attention from dictionary [D].

    Implementation:
      - Soft assignment: take expected body-part weights as the
        weighted sum over all glosses using their predicted probabilities.
      - Compare against the model's implied body-part weighting
        (derived from the feature split: pose=132/1662, face=1404/1662,
        hand=126/1662) as a regularisation target.
      - MSE between predicted distribution and dictionary distribution.

    Note: This is a lightweight proxy distillation.
          Full distillation would require intermediate layer access.
    """
    # Soft gloss probabilities: (B, T, C)  C = num_glosses+1
    probs = tf.nn.softmax(logits, axis=-1)          # already softmax, reuse

    # Build dictionary weight matrix: (num_glosses+1, 3)
    # index 0 = blank, uniform weights
    C = logits.shape[-1] or tf.shape(logits)[-1]
    num_glosses = int(C) - 1

    dict_matrix = np.zeros((num_glosses + 1, 3), dtype=np.float32)
    dict_matrix[0] = [1/3, 1/3, 1/3]       # blank → uniform
    for idx in range(1, num_glosses + 1):
        gloss = idx2gloss.get(idx, '').upper()
        if gloss in att_dict:
            dict_matrix[idx] = att_dict[gloss]
        else:
            dict_matrix[idx] = [1/3, 1/3, 1/3]

    dict_tf = tf.constant(dict_matrix, dtype=tf.float32)   # (C, 3)

    # Expected body-part weights at each frame: (B, T, 3)
    expected_att = tf.matmul(probs, dict_tf)

    # Implied model body-part weighting (fixed proportions from keypoint split)
    # pose=132/1662 ≈ 0.0794,  face=1404/1662 ≈ 0.845,  hand=126/1662 ≈ 0.0758
    implied = tf.constant([[0.0794, 0.8449, 0.0758]], dtype=tf.float32)  # (1, 3)

    # MSE between what dictionary says vs model's fixed proportions
    loss = tf.reduce_mean(tf.square(expected_att - implied))
    return alpha * loss


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY CTC DECODE  (for validation accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_decode(probs_batch: np.ndarray, input_lengths: np.ndarray) -> list[list[int]]:
    """
    Greedy CTC decode: argmax per frame → collapse repeats → remove blank.

    Returns list of decoded label sequences (as index lists).
    """
    blank_idx = probs_batch.shape[-1] - 1   # blank at last index (num_classes-1)
    results = []
    for probs, T in zip(probs_batch, input_lengths):
        indices = np.argmax(probs[:T], axis=-1)               # (T,)
        # Collapse repeats
        collapsed = [indices[0]]
        for idx in indices[1:]:
            if idx != collapsed[-1]:
                collapsed.append(idx)
        # Remove blank (index 0)
        decoded = [i for i in collapsed if i != blank_idx]
        results.append(decoded)
    return results


def sequence_accuracy(preds: list, labels_batch: np.ndarray, label_lengths: np.ndarray) -> float:
    """Exact-match accuracy: prediction must match label exactly."""
    correct = 0
    for pred, label, L in zip(preds, labels_batch, label_lengths):
        gt = list(label[:L])
        if pred == gt:
            correct += 1
    return correct / len(preds)


def _levenshtein_distance(ref: list, hyp: list) -> int:
    """Compute Levenshtein (edit) distance between two sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def word_error_rate(preds: list, labels_batch: np.ndarray, label_lengths: np.ndarray) -> float:
    """
    Word Error Rate (WER) = sum(edit_distances) / sum(reference_lengths).
    Lower is better: 0.0 = perfect, 1.0 = 100% errors.
    """
    total_edits = 0
    total_ref_len = 0
    for pred, label, L in zip(preds, labels_batch, label_lengths):
        gt = [int(x) for x in label[:L]]
        pred_int = [int(x) for x in pred]
        total_edits += _levenshtein_distance(gt, pred_int)
        total_ref_len += max(len(gt), 1)
    return total_edits / max(total_ref_len, 1)


def beam_search_decode(
    probs_batch: np.ndarray,
    input_lengths: np.ndarray,
    beam_width: int = 10,
) -> list[list[int]]:
    """
    CTC beam search decode using TensorFlow's built-in decoder.
    Often recovers 10-20% accuracy compared to greedy decode.
    """
    probs_tf = tf.cast(probs_batch, tf.float32)
    log_probs = tf.math.log(tf.clip_by_value(probs_tf, 1e-10, 1.0))
    log_probs_t = tf.transpose(log_probs, [1, 0, 2])  # (T, B, C)
    seq_lens = tf.cast(input_lengths, tf.int32)

    decoded, _ = tf.nn.ctc_beam_search_decoder(
        log_probs_t, seq_lens, beam_width=beam_width, top_paths=1,
    )

    sparse = decoded[0]
    B = probs_batch.shape[0]
    results = [[] for _ in range(B)]
    for idx, val in zip(sparse.indices.numpy(), sparse.values.numpy()):
        results[int(idx[0])].append(int(val))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────

def train_step(model, optimizer, batch, att_dict_tf, idx2gloss_list, alpha):
    X         = tf.cast(batch['inputs'],        tf.float32)
    labels    = tf.cast(batch['labels'],        tf.int32)
    in_lens   = tf.cast(batch['input_lengths'], tf.int32)
    lbl_lens  = tf.cast(batch['label_lengths'], tf.int32)

    with tf.GradientTape() as tape:
        logits = model(X, training=True)                    # (B, T, C)
        loss_ctc  = ctc_loss_fn(labels, logits, in_lens, lbl_lens)
        loss_dist = alpha * tf.reduce_mean(tf.square(
            tf.reduce_mean(tf.nn.softmax(logits, axis=-1), axis=1)
        ))   # simplified distillation regularisation
        loss = loss_ctc + loss_dist

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, loss_ctc, loss_dist


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Datasets ──────────────────────────────────────────────────────────────
    train_gen, val_gen, meta = build_datasets(
        seq_dir        = args.seq_dir,
        label_path     = args.label_path,
        vocab_path     = args.vocab_path,
        batch_size     = args.batch_size,
        val_split      = args.val_split,
        seed           = args.seed,
        active_only    = args.active_vocab,
        augment_factor = args.augment_factor,
    )
    idx2gloss = meta['idx2gloss']

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_sentence_model(
        num_glosses     = meta['num_glosses'],
        word_model_path = args.word_model,
        freeze_encoder  = args.freeze_encoder,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate = args.lr,
        first_decay_steps     = len(train_gen) * 5,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # ── Checkpointing ──────────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    print(f'\n{"="*60}')
    print(f'TRAINING START  —  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'  Train batches : {len(train_gen)}')
    print(f'  Val batches   : {len(val_gen)}')
    print(f'  Epochs        : {args.epochs}')
    print(f'  Batch size    : {args.batch_size}')
    print(f'  LR            : {args.lr}')
    print(f'  Freeze enc    : {args.freeze_encoder}')
    print(f'  Checkpoint    : {ckpt_dir}')
    print(f'{"="*60}\n')

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_wer': []}

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        train_losses = []
        for step in range(len(train_gen)):
            batch = train_gen[step]
            loss, l_ctc, l_dist = train_step(
                model, optimizer, batch,
                None, None, args.distill_alpha,
            )
            train_losses.append(float(loss))

            if (step + 1) % 10 == 0:
                print(f'  Epoch {epoch:03d} [{step+1:3d}/{len(train_gen)}]  '
                      f'loss={np.mean(train_losses):.4f}  '
                      f'ctc={float(l_ctc):.4f}  dist={float(l_dist):.4f}')

        train_gen.on_epoch_end()
        mean_train = np.mean(train_losses)

        # ── Validation ─────────────────────────────────────────────────────────
        val_losses, val_accs, val_wers = [], [], []
        for step in range(len(val_gen)):
            batch    = val_gen[step]
            X        = tf.cast(batch['inputs'],        tf.float32)
            labels   = batch['labels']
            in_lens  = batch['input_lengths']
            lbl_lens = batch['label_lengths']

            logits   = model(X, training=False)
            v_loss   = ctc_loss_fn(
                tf.cast(labels, tf.int32), logits,
                tf.cast(in_lens, tf.int32), tf.cast(lbl_lens, tf.int32),
            )
            # Beam search decode (better than greedy)
            if args.beam_width > 1:
                preds = beam_search_decode(logits.numpy(), in_lens, args.beam_width)
            else:
                preds = greedy_decode(logits.numpy(), in_lens)
            v_acc  = sequence_accuracy(preds, labels, lbl_lens)
            v_wer  = word_error_rate(preds, labels, lbl_lens)
            val_losses.append(float(v_loss))
            val_accs.append(v_acc)
            val_wers.append(v_wer)

        mean_val  = np.mean(val_losses)
        mean_acc  = np.mean(val_accs)
        mean_wer  = np.mean(val_wers)
        history['train_loss'].append(mean_train)
        history['val_loss'].append(mean_val)
        history['val_acc'].append(mean_acc)
        history['val_wer'].append(mean_wer)

        print(f'\nEpoch {epoch:03d}  '
              f'train={mean_train:.4f}  val={mean_val:.4f}  '
              f'acc={mean_acc:.3f}  WER={mean_wer:.3f}')

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            model.save(str(ckpt_dir / 'best_sentence_model'))
            print(f'  ✓ Saved best model  (val_loss={best_val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break

        print()

    # ── Save history ───────────────────────────────────────────────────────────
    hist_path = ckpt_dir / 'training_history.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'History saved to {hist_path}')
    print(f'Best val loss: {best_val_loss:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir',    default=str(ISL_SEQ_DIR / 'keypoints'))
    parser.add_argument('--label_path', default=str(ISL_SEQ_DIR / 'labels.json'))
    parser.add_argument('--vocab_path', default=str(ACTION_MAPPING_PATH))
    parser.add_argument('--word_model', default=str(WORD_MODEL_PATH))
    parser.add_argument('--ckpt_dir',   default=str(ISL_SEQ_DIR / 'checkpoints' / 'sentence'))
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--val_split',  type=float, default=0.15)
    parser.add_argument('--distill_alpha', type=float, default=0.1)
    parser.add_argument('--patience',   type=int,   default=10)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--beam_width', type=int,   default=10,
                        help='Beam width for CTC decode (1=greedy)')
    parser.add_argument('--active_vocab', action='store_true',
                        help='Reduce vocab to only glosses in labels (~80 vs 409)')
    parser.add_argument('--augment_factor', type=int, default=0,
                        help='Augmentation factor (0=off, 10=10x data)')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder, only train CTC head')
    args = parser.parse_args()

    train(args)
