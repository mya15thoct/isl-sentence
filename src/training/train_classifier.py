"""
Trainer for ISL Sentence Classifier (Approach D).

Simple cross-entropy training — no CTC, no alignment.
Uses model.compile + model.fit with Keras callbacks.

Two-phase option (--freeze_encoder):
  Phase 1 (5 epochs): only train classification head, encoder frozen
  Phase 2 (up to --epochs): unfreeze all, fine-tune with lr/10

Usage (server):
  python src/training/train_classifier.py \\
      --augment_factor 10 \\
      --epochs 100 \\
      --freeze_encoder

Quick test (no GPU):
  python src/training/train_classifier.py --epochs 2 --batch_size 4
"""

import json
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR, WORD_MODEL_PATH

sys.path.append(str(Path(__file__).parent.parent))
from data.classifier_data_loader import build_classifier_datasets
from models.classifier_model import build_classifier_model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_per_class(model, val_gen, id2sentence: dict) -> dict:
    """
    Run inference on val set and return per-class accuracy.
    Useful for identifying which sentences the model struggles with.
    """
    correct_by_class = {}
    total_by_class   = {}

    for i in range(len(val_gen)):
        X, y_true = val_gen[i]
        probs  = model(tf.cast(X, tf.float32), training=False).numpy()
        y_pred = np.argmax(probs, axis=-1)

        for gt, pred in zip(y_true, y_pred):
            total_by_class[gt]   = total_by_class.get(gt, 0) + 1
            correct_by_class[gt] = correct_by_class.get(gt, 0) + int(gt == pred)

    results = {}
    for sid, total in sorted(total_by_class.items()):
        correct  = correct_by_class.get(sid, 0)
        sentence = id2sentence.get(sid, f'[{sid}]')
        results[sentence] = {'correct': correct, 'total': total,
                             'acc': correct / total}

    overall = sum(r['correct'] for r in results.values()) / max(
        sum(r['total'] for r in results.values()), 1)
    results['__overall__'] = {'acc': overall}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Data ──────────────────────────────────────────────────────────────────
    train_gen, val_gen, meta = build_classifier_datasets(
        seq_dir        = args.seq_dir,
        label_path     = args.label_path,
        batch_size     = args.batch_size,
        val_split      = args.val_split,
        seed           = args.seed,
        augment_factor = args.augment_factor,
    )
    num_sentences = meta['num_sentences']
    id2sentence   = meta['id2sentence']

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_classifier_model(
        num_sentences   = num_sentences,
        word_model_path = args.word_model,
        freeze_encoder  = args.freeze_encoder,
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / 'sentence_id_map.json', 'w', encoding='utf-8') as f:
        json.dump({
            'sentence2id':     meta['sentence2id'],
            'sentences_sorted': meta['sentences_sorted'],
        }, f, indent=2, ensure_ascii=False)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=args.patience,
            restore_best_weights=True, mode='max', verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / 'best_classifier'),
            monitor='val_accuracy', save_best_only=True,
            mode='max', verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=10,
            min_lr=1e-6, mode='max', verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(ckpt_dir / 'training_log.csv')),
    ]

    print(f'\n{"="*60}')
    print(f'SENTENCE CLASSIFIER  —  {datetime.now():%Y-%m-%d %H:%M}')
    print(f'  Sentence classes : {num_sentences}')
    print(f'  Train batches    : {len(train_gen)}')
    print(f'  Val   batches    : {len(val_gen)}')
    print(f'  Epochs           : {args.epochs}')
    print(f'  Batch size       : {args.batch_size}')
    print(f'  Augment factor   : {args.augment_factor}×')
    print(f'  Two-phase        : {args.freeze_encoder}')
    print(f'  Checkpoint       : {ckpt_dir}')
    print(f'{"="*60}\n')

    # ── Phase 1: warm up classification head (optional) ───────────────────────
    if args.freeze_encoder:
        print('--- Phase 1: warm up head (encoder frozen, 5 epochs) ---')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=1)

        # Unfreeze all layers for phase 2
        for layer in model.layers:
            layer.trainable = True
        print('\n--- Phase 2: fine-tune all layers ---')
        lr_phase2 = args.lr   # full LR — encoder already warm, no need to reduce
    else:
        lr_phase2 = args.lr

    # ── Phase 2: full training ─────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_phase2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save history ───────────────────────────────────────────────────────────
    hist_path = ckpt_dir / 'training_history.json'
    with open(hist_path, 'w') as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    best_val_acc = max(history.history.get('val_accuracy', [0.0]))
    print(f'\nBest val accuracy : {best_val_acc:.3f}  ({best_val_acc*100:.1f}%)')

    # ── Per-class evaluation ───────────────────────────────────────────────────
    print('\n--- Per-class accuracy on val set ---')
    results = evaluate_per_class(model, val_gen, id2sentence)
    overall = results.pop('__overall__', {}).get('acc', 0.0)

    for sentence, r in sorted(results.items(), key=lambda x: x[1]['acc']):
        mark = '✓' if r['acc'] >= 1.0 else ('~' if r['acc'] > 0 else '✗')
        print(f'  {mark} [{r["correct"]}/{r["total"]}]  {sentence}')

    print(f'\nOverall val accuracy : {overall:.3f}  ({overall*100:.1f}%)')

    results_path = ckpt_dir / 'per_class_accuracy.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({'overall': overall, 'per_class': results}, f,
                  indent=2, ensure_ascii=False)
    print(f'Results saved to {results_path}')


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ISL sentence classifier (Approach D)')

    parser.add_argument('--seq_dir',    default=str(ISL_SEQ_DIR / 'keypoints'),
                        help='Path to extracted .npy sequences folder')
    parser.add_argument('--label_path', default=str(ISL_SEQ_DIR / 'labels.json'),
                        help='Path to labels.json')
    parser.add_argument('--word_model', default=str(WORD_MODEL_PATH),
                        help='Path to best_model_combined (word model)')
    parser.add_argument('--ckpt_dir',
                        default=str(ISL_SEQ_DIR / 'checkpoints' / 'classifier'),
                        help='Directory to save checkpoints')

    parser.add_argument('--epochs',      type=int,   default=300)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--val_split',   type=float, default=0.15)
    parser.add_argument('--patience',    type=int,   default=15,
                        help='EarlyStopping patience (epochs without val_acc improvement)')
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--augment_factor', type=int, default=10,
                        help='Augmentation multiplier (10 = 10× training data)')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Two-phase training: warm up head first, then fine-tune all')

    args = parser.parse_args()
    train(args)
