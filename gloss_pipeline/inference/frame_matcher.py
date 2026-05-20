"""
Frame-level template matching for ISL sentence decoding.

For each class in ISL-Frames-Data, average all keypoints → template.
For each frame in sentence video, find most similar template → gloss.
Decode: sliding window majority vote + consecutive duplicate removal.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FrameMatcher:
    def __init__(
        self,
        kp_dir:        str   = None,
        window_size:   int   = 15,
        stride:        int   = 5,
        conf_threshold: float = 0.6,
    ):
        kp_dir = Path(kp_dir or str(ISL_SEQ_DIR / 'frame_keypoints'))

        self.window_size    = window_size
        self.stride         = stride
        self.conf_threshold = conf_threshold

        # Build templates: mean keypoint per class
        self.templates  = {}   # class_name → (1662,)
        self.class_names = []

        class_dirs = sorted([d for d in kp_dir.iterdir() if d.is_dir()])
        for class_dir in class_dirs:
            npy_files = sorted(class_dir.glob('*.npy'))
            if not npy_files:
                continue
            vecs = []
            for npy in npy_files:
                kp = np.load(npy).astype(np.float32)
                if kp.shape[0] == 1662:
                    vecs.append(kp)
            if vecs:
                self.templates[class_dir.name.upper()] = np.mean(vecs, axis=0)

        self.class_names = sorted(self.templates.keys())
        self.template_matrix = np.stack(
            [self.templates[c] for c in self.class_names])  # (N, 1662)

        # Normalize for fast cosine similarity
        norms = np.linalg.norm(self.template_matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self.template_norm = self.template_matrix / norms

        print(f'[FrameMatcher] {len(self.class_names)} templates loaded from {kp_dir}')

    def _match_frame(self, frame: np.ndarray) -> tuple[str, float]:
        """Single frame (1662,) → (gloss, cosine_similarity)"""
        norm = np.linalg.norm(frame)
        if norm < 1e-8:
            return '__blank__', 0.0
        frame_norm = frame / norm
        sims = self.template_norm @ frame_norm   # (N,)
        idx  = int(np.argmax(sims))
        return self.class_names[idx], float(sims[idx])

    def decode(self, sequence: np.ndarray) -> list[tuple[str, float]]:
        """
        sequence: (T, 1662)
        Returns decoded gloss sequence as [(gloss, confidence), ...]
        """
        T    = sequence.shape[0]
        raw  = []

        for start in range(0, T - self.window_size + 1, self.stride):
            window = sequence[start: start + self.window_size]

            # Majority vote within window
            votes = {}
            for frame in window:
                gloss, conf = self._match_frame(frame)
                if gloss == '__blank__':
                    continue
                if gloss not in votes:
                    votes[gloss] = []
                votes[gloss].append(conf)

            if not votes:
                continue

            # Pick gloss with highest mean confidence
            best_gloss = max(votes, key=lambda g: np.mean(votes[g]))
            best_conf  = float(np.mean(votes[best_gloss]))
            raw.append((best_gloss, best_conf))

        # Filter low confidence
        filtered = [(g, c) for g, c in raw if c >= self.conf_threshold]

        # Remove consecutive duplicates
        decoded = []
        for gloss, conf in filtered:
            if not decoded or decoded[-1][0] != gloss:
                decoded.append((gloss, conf))

        return decoded

    def decode_to_glosses(self, sequence: np.ndarray) -> list[str]:
        return [g for g, _ in self.decode(sequence)]
