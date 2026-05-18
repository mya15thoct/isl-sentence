"""
Sliding Window Inference for ISL Sentence Recognition (Phase 1 — Approach A).

Uses the pretrained word model (best_model_combined) as a sign detector
applied to overlapping windows over continuous keypoint sequences.

Pipeline:
  1. Extract overlapping windows from (T, 1662) keypoint sequence
  2. Pad each window to word model's expected seq_len
  3. Classify each window with word model → (class_idx, confidence)
  4. Post-process: threshold, merge consecutive, remove blanks
  5. Output: ordered gloss sequence

Usage:
  # Single evaluation (default params)
  python src/inference/sliding_window.py

  # Grid search over hyperparameters
  python src/inference/sliding_window.py --grid_search

  # Custom params
  python src/inference/sliding_window.py --window_size 30 --stride 5 --threshold 0.4

  # Save detailed results
  python src/inference/sliding_window.py --save_results /mnt/ngan/ISL-Sequences/sliding_window_results.json
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR, WORD_MODEL_PATH, ACTION_MAPPING_PATH


# ─────────────────────────────────────────────────────────────────────────────
# LEVENSHTEIN DISTANCE + LCS (for WER & per-word matching)
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(ref: list, hyp: list) -> int:
    """Levenshtein (edit) distance between two sequences."""
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


def _lcs_match(ref: list, hyp: list) -> list[bool]:
    """
    Longest Common Subsequence matching.
    Returns a bool list (same length as ref) indicating which GT words
    were matched in the prediction (order-aware).

    Example:
        ref = ['I', 'HELP', 'YOU'],  hyp = ['I', 'WATER', 'YOU']
        → [True, False, True]   (I ✓, HELP ✗, YOU ✓)
    """
    n, m = len(ref), len(hyp)
    # Build LCS table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find which ref items matched
    matched = [False] * n
    i, j = n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1]:
            matched[i - 1] = True
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING WINDOW RECOGNIZER
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowRecognizer:
    """
    Recognize ISL sentence by sliding a window over keypoint sequence
    and classifying each window with the pretrained word model.

    The word model (best_model_combined) was trained on isolated signs
    with shape (seq_len, 1662). We extract overlapping windows from
    continuous sentence video and classify each window independently.
    """

    def __init__(
        self,
        word_model_path: str,
        action_mapping_path: str,
        window_size: int = 30,
        stride: int = 5,
        confidence_threshold: float = 0.4,
        seq_len: int = None,
    ):
        """
        Args:
            word_model_path:      Path to best_model_combined (SavedModel)
            action_mapping_path:  Path to action_mapping_combined.json
            window_size:          Number of frames per window (~1s at 30fps)
            stride:               Step between consecutive windows
            confidence_threshold: Min softmax confidence to accept a prediction
            seq_len:              Override model's expected input length (auto-detect if None)
        """
        # Load word model
        print(f'[SlidingWindow] Loading word model from: {word_model_path}')
        self.model = tf.keras.models.load_model(str(word_model_path))

        # Auto-detect seq_len from model input shape
        input_shape = self.model.input_shape
        if seq_len is not None:
            self.seq_len = seq_len
        elif input_shape[1] is not None:
            self.seq_len = input_shape[1]
        else:
            self.seq_len = window_size
        print(f'[SlidingWindow] Model input: {input_shape}, seq_len={self.seq_len}')

        # Load action mapping (idx → gloss name)
        with open(action_mapping_path, encoding='utf-8') as f:
            mapping = json.load(f)
        self.idx2gloss = {int(k): v.upper() for k, v in mapping.items()}
        self.num_classes = len(self.idx2gloss)
        print(f'[SlidingWindow] Vocab: {self.num_classes} classes')

        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self._debug_count = 0  # tracks how many samples have been debug-printed

    def extract_windows(self, keypoints: np.ndarray) -> list[np.ndarray]:
        """
        Extract overlapping windows from (T, 1662) keypoint sequence.

        If T <= window_size, returns the entire sequence as a single window.
        Otherwise, slides with given stride and ensures last frames are covered.
        """
        T = keypoints.shape[0]
        windows = []

        if T <= self.window_size:
            windows.append(keypoints)
        else:
            for start in range(0, T - self.window_size + 1, self.stride):
                windows.append(keypoints[start:start + self.window_size])

            # Ensure tail coverage if last window doesn't reach the end
            last_start = (T - self.window_size)
            if last_start % self.stride != 0:
                windows.append(keypoints[last_start:T])

        return windows

    def pad_window(self, window: np.ndarray) -> np.ndarray:
        """Pad (or truncate center) a window to match word model's seq_len."""
        T = window.shape[0]
        if T == self.seq_len:
            return window
        elif T < self.seq_len:
            padded = np.zeros((self.seq_len, 1662), dtype=np.float32)
            padded[:T] = window
            return padded
        else:
            start = (T - self.seq_len) // 2
            return window[start:start + self.seq_len]

    def classify_windows(
        self, windows: list[np.ndarray], debug: bool = False,
    ) -> list[tuple[int, float, str]]:
        """
        Classify each window with word model in a single batch.

        Returns:
            List of (class_idx, confidence, gloss_name) per window.
        """
        batch = np.array(
            [self.pad_window(w) for w in windows], dtype=np.float32,
        )
        probs = self.model.predict(batch, verbose=0)  # (N, num_classes)

        results = []
        for i, p in enumerate(probs):
            idx = int(np.argmax(p))
            conf = float(p[idx])
            gloss = self.idx2gloss.get(idx, f'UNK_{idx}')
            results.append((idx, conf, gloss))

        # Debug: print confidence stats + top-3 for first few samples
        if debug and self._debug_count < 3:
            confs = [r[1] for r in results]
            top3_per_window = []
            for p in probs[:3]:  # first 3 windows
                top3_idx = np.argsort(p)[-3:][::-1]
                top3 = [(self.idx2gloss.get(int(j), '?'), f'{p[j]:.4f}') for j in top3_idx]
                top3_per_window.append(top3)
            print(f'    [DEBUG] {len(windows)} windows | '
                  f'conf: min={min(confs):.4f} max={max(confs):.4f} mean={np.mean(confs):.4f}')
            for wi, t3 in enumerate(top3_per_window):
                print(f'    [DEBUG] window[{wi}] top-3: {t3}')
            self._debug_count += 1

        return results

    def post_process(
        self,
        predictions: list[tuple[int, float, str]],
    ) -> list[str]:
        """
        Convert raw window predictions into a clean gloss sequence.

        Steps:
          1. Filter by confidence threshold (low → BLANK)
          2. Merge consecutive identical predictions
          3. Remove BLANKs
        """
        # 1. Threshold
        filtered = []
        for _, conf, gloss in predictions:
            filtered.append(gloss if conf >= self.confidence_threshold else '__BLANK__')

        if not filtered:
            return []

        # 2. Merge consecutive duplicates
        merged = [filtered[0]]
        for g in filtered[1:]:
            if g != merged[-1]:
                merged.append(g)

        # 3. Remove blanks
        return [g for g in merged if g != '__BLANK__']

    def recognize(self, keypoints: np.ndarray, debug: bool = False) -> dict:
        """
        Full pipeline: (T, 1662) keypoints → gloss sequence.

        Returns dict with:
          glosses:          list[str]  — predicted gloss names
          raw_predictions:  list[tuple] — per-window (idx, conf, gloss)
          num_windows:      int
        """
        windows = self.extract_windows(keypoints)
        raw_preds = self.classify_windows(windows, debug=debug)
        glosses = self.post_process(raw_preds)

        return {
            'glosses': glosses,
            'raw_predictions': raw_preds,
            'num_windows': len(windows),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER FINDER  (reused from data_loader.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def _find_folder(seq_dir: Path, sentence: str) -> Path | None:
    """Find sequence folder for a sentence, case-insensitive."""
    target = sentence.strip().lower().replace(' ', '_')

    if (seq_dir / target).exists():
        return seq_dir / target

    target_norm = target.replace('_', '')
    for folder in seq_dir.iterdir():
        if folder.is_dir() and folder.name.lower().replace('_', '') == target_norm:
            return folder
    return None


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_dataset(
    recognizer: SlidingWindowRecognizer,
    seq_dir: str,
    label_path: str,
    verbose: bool = True,
    debug: bool = False,
) -> dict:
    """
    Evaluate sliding window recognizer on the full sentence dataset.

    Args:
        recognizer:  Initialised SlidingWindowRecognizer
        seq_dir:     Path to sequences/ folder with .npy files
        label_path:  Path to labels.json
        verbose:     Print per-sample results

    Returns:
        dict with 'wer', 'exact_accuracy', 'total_samples', 'details'
    """
    seq_dir = Path(seq_dir)

    with open(label_path, encoding='utf-8') as f:
        labels = json.load(f)

    total_edits = 0
    total_ref_len = 0
    exact_matches = 0
    total_samples = 0
    results_detail = []

    # Per-gloss tracking: {GLOSS: {'total': N, 'correct': N}}
    gloss_stats = {}

    for sentence, gt_glosses in labels.items():
        gt_upper = [g.upper() for g in gt_glosses]
        folder = _find_folder(seq_dir, sentence)
        if folder is None:
            continue

        for npy_file in sorted(folder.glob('*.npy')):
            keypoints = np.load(npy_file)
            result = recognizer.recognize(keypoints, debug=debug)
            pred = result['glosses']

            edits = _levenshtein(gt_upper, pred)
            ref_len = max(len(gt_upper), 1)
            sample_wer = edits / ref_len
            is_exact = (pred == gt_upper)

            # Per-word matching via LCS
            word_matches = _lcs_match(gt_upper, pred)

            # Track per-gloss recall
            for gloss, hit in zip(gt_upper, word_matches):
                if gloss not in gloss_stats:
                    gloss_stats[gloss] = {'total': 0, 'correct': 0}
                gloss_stats[gloss]['total'] += 1
                if hit:
                    gloss_stats[gloss]['correct'] += 1

            total_edits += edits
            total_ref_len += ref_len
            total_samples += 1
            if is_exact:
                exact_matches += 1

            # Build per-word annotation: "I✓ HELP✗ YOU✓"
            word_annotations = []
            for g, hit in zip(gt_upper, word_matches):
                word_annotations.append(f'{g}✓' if hit else f'{g}✗')

            results_detail.append({
                'sentence': sentence,
                'file': npy_file.name,
                'ground_truth': gt_upper,
                'predicted': pred,
                'word_matches': word_annotations,
                'words_correct': sum(word_matches),
                'words_total': len(gt_upper),
                'wer': round(sample_wer, 4),
                'exact_match': is_exact,
                'num_windows': result['num_windows'],
            })

            if verbose:
                mark = '✓' if is_exact else '✗'
                word_str = '  '.join(word_annotations)
                print(f'  {mark} [{npy_file.name}]  [{word_str}]  →  Pred={pred}  '
                      f'(WER={sample_wer:.2f}  words={sum(word_matches)}/{len(gt_upper)})')

    overall_wer = total_edits / max(total_ref_len, 1)
    exact_acc = exact_matches / max(total_samples, 1)

    # Compute per-gloss recall and sort
    gloss_recall = []
    for gloss, stats in gloss_stats.items():
        recall = stats['correct'] / max(stats['total'], 1)
        gloss_recall.append({
            'gloss': gloss,
            'correct': stats['correct'],
            'total': stats['total'],
            'recall': round(recall, 4),
        })
    gloss_recall.sort(key=lambda x: (-x['recall'], -x['total']))

    total_word_correct = sum(s['correct'] for s in gloss_stats.values())
    total_word_count = sum(s['total'] for s in gloss_stats.values())
    word_accuracy = total_word_correct / max(total_word_count, 1)

    # ── Summary ──
    print(f'\n{"=" * 60}')
    print(f'SLIDING WINDOW EVALUATION')
    print(f'{"=" * 60}')
    print(f'  Samples          : {total_samples}')
    print(f'  Window / Stride  : {recognizer.window_size} / {recognizer.stride}')
    print(f'  Threshold        : {recognizer.confidence_threshold}')
    print(f'  ────────────────────────────────')
    print(f'  WER              : {overall_wer:.4f}  ({overall_wer*100:.1f}%)')
    print(f'  Word Accuracy    : {word_accuracy:.4f}  ({total_word_correct}/{total_word_count})')
    print(f'  Exact-match      : {exact_acc:.4f}  ({exact_matches}/{total_samples})')

    # ── Per-gloss recall table ──
    print(f'\n  {"─" * 50}')
    print(f'  PER-GLOSS RECALL  (top 15 best + bottom 15 worst)')
    print(f'  {"─" * 50}')
    print(f'  {"Gloss":<25s} {"Correct":>7s} {"Total":>6s} {"Recall":>7s}')
    print(f'  {"─" * 50}')

    # Top 15
    for g in gloss_recall[:15]:
        bar = '█' * int(g['recall'] * 10)
        print(f'  {g["gloss"]:<25s} {g["correct"]:>7d} {g["total"]:>6d} {g["recall"]:>6.1%}  {bar}')

    if len(gloss_recall) > 30:
        print(f'  {"...":^50s}')

    # Bottom 15
    bottom = gloss_recall[-15:] if len(gloss_recall) > 15 else []
    for g in bottom:
        if g in gloss_recall[:15]:
            continue  # skip if already shown
        bar = '█' * int(g['recall'] * 10)
        print(f'  {g["gloss"]:<25s} {g["correct"]:>7d} {g["total"]:>6d} {g["recall"]:>6.1%}  {bar}')

    print(f'  {"─" * 50}')
    n_perfect = sum(1 for g in gloss_recall if g['recall'] == 1.0)
    n_zero = sum(1 for g in gloss_recall if g['recall'] == 0.0)
    print(f'  Unique glosses: {len(gloss_recall)}  |  '
          f'100% recall: {n_perfect}  |  0% recall: {n_zero}')
    print(f'{"=" * 60}')

    return {
        'wer': overall_wer,
        'word_accuracy': word_accuracy,
        'exact_accuracy': exact_acc,
        'total_samples': total_samples,
        'exact_matches': exact_matches,
        'gloss_recall': gloss_recall,
        'details': results_detail,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(
    seq_dir: str,
    label_path: str,
    word_model_path: str,
    action_mapping_path: str,
) -> tuple[dict, list]:
    """
    Grid search over (window_size, stride, threshold) to find best config.

    Loads the word model once, then reuses for all configs.
    Returns (best_config, all_results).
    """
    window_sizes = [20, 25, 30, 40, 50]
    strides = [3, 5, 8, 10]
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

    print('[GridSearch] Loading word model (once)...')
    model = tf.keras.models.load_model(str(word_model_path))

    with open(action_mapping_path, encoding='utf-8') as f:
        mapping = json.load(f)
    idx2gloss = {int(k): v.upper() for k, v in mapping.items()}

    input_shape = model.input_shape
    seq_len = input_shape[1] if input_shape[1] is not None else 30

    n_configs = len(window_sizes) * len(strides) * len(thresholds)
    print(f'[GridSearch] Testing {n_configs} configurations...\n')

    best_wer = float('inf')
    best_config = {}
    all_results = []

    for W in window_sizes:
        for S in strides:
            if S >= W:
                continue  # stride must be < window
            for thresh in thresholds:
                # Build recognizer without reloading model
                rec = SlidingWindowRecognizer.__new__(SlidingWindowRecognizer)
                rec.model = model
                rec.idx2gloss = idx2gloss
                rec.num_classes = len(idx2gloss)
                rec.seq_len = seq_len
                rec.window_size = W
                rec.stride = S
                rec.confidence_threshold = thresh

                result = evaluate_on_dataset(rec, seq_dir, label_path, verbose=False)

                entry = {
                    'window_size': W, 'stride': S, 'threshold': thresh,
                    'wer': round(result['wer'], 4),
                    'exact_accuracy': round(result['exact_accuracy'], 4),
                }
                all_results.append(entry)

                if result['wer'] < best_wer:
                    best_wer = result['wer']
                    best_config = entry.copy()

                print(f'  W={W:2d}  S={S:2d}  thresh={thresh:.1f}  '
                      f'→  WER={result["wer"]:.3f}  acc={result["exact_accuracy"]:.3f}')

    print(f'\n{"=" * 60}')
    print(f'BEST CONFIG:  W={best_config["window_size"]}  S={best_config["stride"]}  '
          f'thresh={best_config["threshold"]}')
    print(f'  →  WER={best_wer:.4f}  ({best_wer*100:.1f}%)')
    print(f'{"=" * 60}')

    return best_config, all_results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sliding Window ISL Sentence Recognition (Phase 1)',
    )
    parser.add_argument('--seq_dir',    default=str(ISL_SEQ_DIR / 'keypoints'))
    parser.add_argument('--label_path', default=str(ISL_SEQ_DIR / 'labels.json'))
    parser.add_argument('--word_model', default=str(WORD_MODEL_PATH))
    parser.add_argument('--action_mapping', default=str(ACTION_MAPPING_PATH))
    parser.add_argument('--window_size', type=int,   default=30)
    parser.add_argument('--stride',      type=int,   default=5)
    parser.add_argument('--threshold',   type=float, default=0.1)
    parser.add_argument('--grid_search', action='store_true',
                        help='Run grid search over hyperparameters')
    parser.add_argument('--debug', action='store_true',
                        help='Print raw confidence stats for first 3 samples')
    parser.add_argument('--save_results', type=str,  default=None,
                        help='Path to save results JSON')
    args = parser.parse_args()

    print(f'\n{"=" * 60}')
    print(f'ISL SLIDING WINDOW RECOGNITION — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'{"=" * 60}\n')

    if args.grid_search:
        best_config, all_results = grid_search(
            seq_dir=args.seq_dir,
            label_path=args.label_path,
            word_model_path=args.word_model,
            action_mapping_path=args.action_mapping,
        )
        if args.save_results:
            save_path = Path(args.save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({
                    'best': best_config,
                    'all': all_results,
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2)
            print(f'Grid search results saved to {save_path}')
    else:
        recognizer = SlidingWindowRecognizer(
            word_model_path=args.word_model,
            action_mapping_path=args.action_mapping,
            window_size=args.window_size,
            stride=args.stride,
            confidence_threshold=args.threshold,
        )

        result = evaluate_on_dataset(
            recognizer,
            seq_dir=args.seq_dir,
            label_path=args.label_path,
            debug=args.debug,
        )

        if args.save_results:
            save_path = Path(args.save_results)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({
                    'config': {
                        'window_size': args.window_size,
                        'stride': args.stride,
                        'threshold': args.threshold,
                    },
                    'wer': result['wer'],
                    'exact_accuracy': result['exact_accuracy'],
                    'total_samples': result['total_samples'],
                    'exact_matches': result['exact_matches'],
                    'details': result['details'],
                    'timestamp': datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
            print(f'Results saved to {save_path}')
