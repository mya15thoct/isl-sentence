"""
Map predicted gloss sequence → natural language sentence.

Strategy: find the sentence in labels.json whose gloss sequence
best matches the predicted glosses using token overlap (Jaccard score).
Falls back to returning the raw gloss sequence if no good match found.
"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR

LABELS_PATH = ISL_SEQ_DIR / 'labels.json'


class GlossToSentence:
    def __init__(self, labels_path: str = None):
        path = Path(labels_path or str(LABELS_PATH))
        with open(path, encoding='utf-8') as f:
            self.labels = json.load(f)   # {sentence: [gloss, ...]}

        # Precompute gloss sets per sentence for fast matching
        self.sentence_glosses = {
            sent: [g.upper() for g in glosses]
            for sent, glosses in self.labels.items()
        }

    def _score(self, predicted: list[str], reference: list[str]) -> float:
        """Token overlap score: F1 between predicted and reference gloss sets."""
        pred_set = set(predicted)
        ref_set  = set(reference)
        if not pred_set or not ref_set:
            return 0.0
        intersection = pred_set & ref_set
        precision = len(intersection) / len(pred_set)
        recall    = len(intersection) / len(ref_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def match(self, glosses: list[str], threshold: float = 0.3) -> tuple[str, float]:
        """
        Find best matching sentence for given gloss sequence.

        Returns:
            (sentence, score) — sentence is raw gloss string if no match above threshold
        """
        best_sent  = None
        best_score = 0.0

        for sent, ref_glosses in self.sentence_glosses.items():
            score = self._score(glosses, ref_glosses)
            if score > best_score:
                best_score = score
                best_sent  = sent

        if best_score < threshold:
            return ' '.join(glosses).lower(), best_score

        return best_sent, best_score

    def __call__(self, glosses: list[str]) -> str:
        sentence, score = self.match(glosses)
        return sentence
