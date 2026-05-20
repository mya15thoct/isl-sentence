"""
Evaluate gloss pipeline on all sentence samples.

Metrics:
  - Gloss precision / recall / F1   (predicted vs expected glosses)
  - Sentence accuracy               (matched sentence == ground truth)

Usage:
  python gloss_pipeline/inference/evaluate.py \
      > /mnt/ngan/ISL-Sequences/logs/gloss_eval.log 2>&1
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR
from gloss_pipeline.inference.frame_matcher import FrameMatcher
from gloss_pipeline.inference.gloss_to_sentence import GlossToSentence


def token_f1(predicted: list, reference: list) -> tuple[float, float, float]:
    pred_set = set(predicted)
    ref_set  = set(reference)
    if not pred_set or not ref_set:
        return 0.0, 0.0, 0.0
    tp        = len(pred_set & ref_set)
    precision = tp / len(pred_set)
    recall    = tp / len(ref_set)
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)
    return precision, recall, f1


def evaluate(
    keypoints_dir: Path = None,
    labels_path:   Path = None,
    window_size:   int  = 30,
    stride:        int  = 10,
    conf_threshold: float = 0.3,
):
    keypoints_dir = keypoints_dir or ISL_SEQ_DIR / 'keypoints'
    labels_path   = labels_path   or ISL_SEQ_DIR / 'labels.json'

    with open(labels_path, encoding='utf-8') as f:
        labels = json.load(f)   # {sentence: [gloss, ...]}

    # folder name → sentence mapping
    folder2sent = {}
    for sent in labels:
        folder = sent.lower().replace(' ', '_').replace(',', '').replace('.', '').replace("'", '')
        folder2sent[folder] = sent

    decoder = FrameMatcher(
        window_size=window_size, stride=stride, conf_threshold=conf_threshold)
    g2s     = GlossToSentence(str(labels_path))

    all_precision, all_recall, all_f1 = [], [], []
    sent_correct = 0
    sent_total   = 0
    per_sentence = {}

    sent_dirs = sorted([d for d in keypoints_dir.iterdir() if d.is_dir()])
    for sent_dir in sent_dirs:
        folder_name = sent_dir.name.lower()
        ground_truth_sent = folder2sent.get(folder_name)
        if ground_truth_sent is None:
            continue
        ref_glosses = [g.upper() for g in labels[ground_truth_sent]]

        npy_files = sorted(sent_dir.glob('*.npy'))
        if not npy_files:
            continue

        sample_precisions, sample_recalls, sample_f1s = [], [], []
        sample_correct = 0

        for npy in npy_files:
            seq = np.load(npy).astype(np.float32)
            pred_glosses = decoder.decode_to_glosses(seq)
            matched_sent, score = g2s.match(pred_glosses)

            p, r, f = token_f1(pred_glosses, ref_glosses)
            sample_precisions.append(p)
            sample_recalls.append(r)
            sample_f1s.append(f)

            if matched_sent == ground_truth_sent:
                sample_correct += 1

        avg_p  = float(np.mean(sample_precisions))
        avg_r  = float(np.mean(sample_recalls))
        avg_f  = float(np.mean(sample_f1s))
        sacc   = sample_correct / len(npy_files)

        all_precision.append(avg_p)
        all_recall.append(avg_r)
        all_f1.append(avg_f)
        sent_correct += sample_correct
        sent_total   += len(npy_files)

        per_sentence[ground_truth_sent] = {
            'precision': avg_p, 'recall': avg_r, 'f1': avg_f,
            'sent_acc': sacc, 'n_samples': len(npy_files),
        }

    # ── Report ─────────────────────────────────────────────────────────────────
    print('\n' + '='*65)
    print('GLOSS PIPELINE EVALUATION')
    print('='*65)
    print(f'{"Sentence":<45} {"P":>5} {"R":>5} {"F1":>5} {"Acc":>5}')
    print('-'*65)

    for sent, r in sorted(per_sentence.items(), key=lambda x: x[1]['f1']):
        mark = '✓' if r['sent_acc'] >= 1.0 else ('~' if r['sent_acc'] > 0 else '✗')
        print(f'{mark} {sent:<43} {r["precision"]:>5.2f} {r["recall"]:>5.2f} '
              f'{r["f1"]:>5.2f} {r["sent_acc"]:>5.2f}')

    print('='*65)
    print(f'  Avg gloss precision : {np.mean(all_precision):.3f}')
    print(f'  Avg gloss recall    : {np.mean(all_recall):.3f}')
    print(f'  Avg gloss F1        : {np.mean(all_f1):.3f}')
    print(f'  Sentence accuracy   : {sent_correct}/{sent_total} '
          f'({sent_correct/max(sent_total,1)*100:.1f}%)')
    print('='*65)

    out_path = ISL_SEQ_DIR / 'logs' / 'gloss_eval.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'avg_precision': float(np.mean(all_precision)),
            'avg_recall':    float(np.mean(all_recall)),
            'avg_f1':        float(np.mean(all_f1)),
            'sentence_accuracy': sent_correct / max(sent_total, 1),
            'per_sentence': per_sentence,
        }, f, indent=2, ensure_ascii=False)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size',   type=int,   default=15)
    parser.add_argument('--stride',        type=int,   default=5)
    parser.add_argument('--threshold',     type=float, default=0.6)
    args = parser.parse_args()
    evaluate(window_size=args.window_size, stride=args.stride,
             conf_threshold=args.threshold)
