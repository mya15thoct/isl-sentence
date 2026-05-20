"""
MLP frame-by-frame decoder for sentence inference.

Two decoding strategies:
  1. sliding_window: majority vote per window (original)
  2. peak_detection: smooth probability curve → find peaks per class (better)
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR


class MLPDecoder:
    def __init__(
        self,
        model_path:     str   = None,
        mapping_path:   str   = None,
        conf_threshold: float = 0.35,
        smooth_sigma:   float = 3.0,
        min_peak_dist:  int   = 10,
        # sliding window params (kept for compatibility)
        window_size:    int   = 15,
        stride:         int   = 5,
    ):
        model_path   = model_path   or str(ISL_SEQ_DIR / 'checkpoints' / 'static_classifier' / 'best_static_classifier')
        mapping_path = mapping_path or str(ISL_SEQ_DIR / 'checkpoints' / 'static_classifier' / 'class_mapping.json')

        self.conf_threshold = conf_threshold
        self.smooth_sigma   = smooth_sigma
        self.min_peak_dist  = min_peak_dist
        self.window_size    = window_size
        self.stride         = stride

        self.model = tf.keras.models.load_model(model_path)

        with open(mapping_path, encoding='utf-8') as f:
            mapping = json.load(f)
        self.id2class = {int(k): v.upper() for k, v in mapping['id2class'].items()}
        print(f'[MLPDecoder] {len(self.id2class)} classes loaded')

    def decode(self, sequence: np.ndarray) -> list[tuple[str, float]]:
        """
        sequence: (T, 1662)

        Steps:
          1. MLP → (T, C) per-frame probabilities
          2. Gaussian smooth each class over time
          3. Find peaks in smoothed probability curves
          4. At each peak position, take the class with highest smoothed prob
          5. Sort peaks by time position → gloss sequence
        """
        # 1. Predict all frames at once
        probs = self.model(
            tf.cast(sequence, tf.float32), training=False).numpy()   # (T, C)

        # 2. Gaussian smooth each class over time
        smoothed = gaussian_filter1d(probs, sigma=self.smooth_sigma, axis=0)  # (T, C)

        # 3. Find peaks per class
        peak_events = []   # (frame_idx, class_idx, confidence)
        for c in range(smoothed.shape[1]):
            peaks, props = find_peaks(
                smoothed[:, c],
                height=self.conf_threshold,
                distance=self.min_peak_dist,
            )
            for p in peaks:
                peak_events.append((int(p), c, float(smoothed[p, c])))

        if not peak_events:
            return []

        # 4. Sort by time position
        peak_events.sort(key=lambda x: x[0])

        # 5. Merge nearby peaks — keep highest confidence within min_peak_dist frames
        merged = []
        for frame_idx, c, conf in peak_events:
            if merged and abs(frame_idx - merged[-1][0]) < self.min_peak_dist:
                if conf > merged[-1][2]:
                    merged[-1] = (frame_idx, c, conf)
            else:
                merged.append((frame_idx, c, conf))

        # 6. Convert to gloss sequence, remove consecutive duplicates
        decoded = []
        for _, c, conf in merged:
            gloss = self.id2class.get(c, '?')
            if not decoded or decoded[-1][0] != gloss:
                decoded.append((gloss, conf))

        return decoded

    def decode_to_glosses(self, sequence: np.ndarray) -> list[str]:
        return [g for g, _ in self.decode(sequence)]
