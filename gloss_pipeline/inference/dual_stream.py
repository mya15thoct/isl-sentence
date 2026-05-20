"""
Dual-stream sliding window inference.

For each window over the keypoint sequence:
  - Word model (BiLSTM)   → top gloss + confidence
  - Static model (MLP)    → top gloss + confidence
  → Keep whichever has higher confidence

Then decode: remove low-confidence windows, merge consecutive duplicates.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR, WORD_MODEL_PATH, ACTION_MAPPING_PATH


class DualStreamDecoder:
    def __init__(
        self,
        word_model_path:   str = None,
        static_model_path: str = None,
        word_mapping_path: str = None,
        static_mapping_path: str = None,
        window_size:  int   = 30,
        stride:       int   = 10,
        conf_threshold: float = 0.3,
    ):
        word_model_path    = word_model_path   or str(WORD_MODEL_PATH)
        static_model_path  = static_model_path or str(
            ISL_SEQ_DIR / 'checkpoints' / 'static_classifier' / 'best_static_classifier')
        word_mapping_path  = word_mapping_path or str(ACTION_MAPPING_PATH)
        static_mapping_path = static_mapping_path or str(
            ISL_SEQ_DIR / 'checkpoints' / 'static_classifier' / 'class_mapping.json')

        self.window_size    = window_size
        self.stride         = stride
        self.conf_threshold = conf_threshold

        print('[DualStream] Loading word model...')
        self.word_model = tf.keras.models.load_model(word_model_path)

        print('[DualStream] Loading static classifier...')
        self.static_model = tf.keras.models.load_model(static_model_path)

        with open(word_mapping_path, encoding='utf-8') as f:
            mapping = json.load(f)
        self.word_id2gloss = {int(k): v.upper() for k, v in mapping.items()}

        with open(static_mapping_path, encoding='utf-8') as f:
            mapping = json.load(f)
        self.static_id2gloss = {int(k): v.upper()
                                 for k, v in mapping['id2class'].items()}

        print(f'[DualStream] Word vocab: {len(self.word_id2gloss)}  '
              f'Static vocab: {len(self.static_id2gloss)}')

    def _predict_word(self, window: np.ndarray):
        """window: (T, 1662) → (gloss, confidence)"""
        expected = self.word_model.input_shape[1]   # fixed seq_len (e.g. 201)
        if expected is not None and window.shape[0] != expected:
            padded = np.zeros((expected, 1662), dtype=np.float32)
            padded[:min(window.shape[0], expected)] = window[:expected]
            window = padded
        x    = window[np.newaxis].astype(np.float32)
        prob = self.word_model.predict(x, verbose=0)[0]
        idx  = int(np.argmax(prob))
        return self.word_id2gloss.get(idx, '?'), float(prob[idx])

    def _predict_static(self, frame: np.ndarray):
        """frame: (1662,) → (gloss, confidence)"""
        x    = frame[np.newaxis].astype(np.float32)    # (1, 1662)
        prob = self.static_model.predict(x, verbose=0)[0]
        idx  = int(np.argmax(prob))
        return self.static_id2gloss.get(idx, '?'), float(prob[idx])

    def decode(self, sequence: np.ndarray) -> list[tuple[str, float]]:
        """
        sequence: (T, 1662)
        Returns list of (gloss, confidence) after decoding.
        """
        T = sequence.shape[0]
        raw = []   # (gloss, confidence) per window

        for start in range(0, T - self.window_size + 1, self.stride):
            window = sequence[start: start + self.window_size]
            mid    = window[len(window) // 2]

            word_gloss,   word_conf   = self._predict_word(window)
            static_gloss, static_conf = self._predict_static(mid)

            if word_conf >= static_conf:
                gloss, conf = word_gloss, word_conf
            else:
                gloss, conf = static_gloss, static_conf

            raw.append((gloss, conf))

        # Filter low confidence
        filtered = [(g, c) for g, c in raw if c >= self.conf_threshold]

        # Merge consecutive duplicates
        decoded = []
        for gloss, conf in filtered:
            if not decoded or decoded[-1][0] != gloss:
                decoded.append((gloss, conf))

        return decoded

    def decode_to_glosses(self, sequence: np.ndarray) -> list[str]:
        return [g for g, _ in self.decode(sequence)]
