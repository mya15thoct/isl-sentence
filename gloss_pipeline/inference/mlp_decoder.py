"""
MLP frame-by-frame decoder for sentence inference.

Apply trained static MLP classifier to each frame of sentence video.
Temporal smoothing via sliding window majority vote → gloss sequence.
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR


class MLPDecoder:
    def __init__(
        self,
        model_path:     str   = None,
        mapping_path:   str   = None,
        window_size:    int   = 15,
        stride:         int   = 5,
        conf_threshold: float = 0.4,
    ):
        model_path   = model_path   or str(ISL_SEQ_DIR / 'checkpoints' / 'static_classifier' / 'best_static_classifier')
        mapping_path = mapping_path or str(ISL_SEQ_DIR / 'checkpoints' / 'static_classifier' / 'class_mapping.json')

        self.window_size    = window_size
        self.stride         = stride
        self.conf_threshold = conf_threshold

        self.model = tf.keras.models.load_model(model_path)

        with open(mapping_path, encoding='utf-8') as f:
            mapping = json.load(f)
        self.id2class = {int(k): v.upper() for k, v in mapping['id2class'].items()}
        print(f'[MLPDecoder] {len(self.id2class)} classes loaded')

    def decode(self, sequence: np.ndarray) -> list[tuple[str, float]]:
        """sequence: (T, 1662) → [(gloss, conf), ...]"""
        T = sequence.shape[0]

        # Batch predict all frames at once
        probs = self.model(
            tf.cast(sequence, tf.float32), training=False).numpy()  # (T, C)

        raw = []
        for start in range(0, T - self.window_size + 1, self.stride):
            window_probs = probs[start: start + self.window_size]   # (W, C)

            # Mean probability over window → majority vote
            mean_prob  = window_probs.mean(axis=0)                  # (C,)
            idx        = int(np.argmax(mean_prob))
            best_gloss = self.id2class.get(idx, '?')
            best_conf  = float(mean_prob[idx])
            raw.append((best_gloss, best_conf))

        # Filter + remove consecutive duplicates
        decoded = []
        for gloss, conf in raw:
            if conf < self.conf_threshold:
                continue
            if not decoded or decoded[-1][0] != gloss:
                decoded.append((gloss, conf))

        return decoded

    def decode_to_glosses(self, sequence: np.ndarray) -> list[str]:
        return [g for g, _ in self.decode(sequence)]
