"""
WordEncoder: load mlp/best_model, extract per-frame bilstm2 features (64-dim).

Architecture up to bilstm2:
  Input (201, 1662) → MLP branches → BiLSTM(64) → BiLSTM(32) → (201, 64)

Usage:
  enc = WordEncoder('/mnt/ngan/recognition/checkpoints/mlp/best_model')
  feats = enc.encode(seq)   # seq: (T, 1662) → (T, 64)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path


class WordEncoder:
    SEQ_LEN = 201   # fixed input length the model was trained with

    def __init__(self, model_path: str):
        full_model = tf.keras.models.load_model(model_path, compile=False)

        # Sub-model: input → bilstm2 output  (None, 201, 64)
        self._encoder = tf.keras.Model(
            inputs  = full_model.input,
            outputs = full_model.get_layer('bilstm2').output,
            name    = 'bilstm2_encoder',
        )
        self._encoder.trainable = False
        print(f'[WordEncoder] loaded  input={self._encoder.input_shape}  '
              f'output={self._encoder.output_shape}')

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        """
        sequence : (T, 1662)  — raw keypoint sequence, any length
        returns  : (T, 64)    — per-frame BiLSTM features (trimmed to T)
        """
        T = len(sequence)
        padded = self._pad(sequence)                          # (201, 1662)
        feats  = self._encoder(
            padded[None].astype(np.float32), training=False
        ).numpy()[0]                                          # (201, 64)
        return feats[:T]                                      # (T, 64)

    def encode_batch(self, sequences: list[np.ndarray],
                     batch_size: int = 32) -> list[np.ndarray]:
        """Encode a list of variable-length sequences efficiently."""
        results = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            Ts    = [len(s) for s in batch]
            padded = np.stack([self._pad(s) for s in batch]).astype(np.float32)
            feats  = self._encoder(padded, training=False).numpy()  # (B, 201, 64)
            for j, T in enumerate(Ts):
                results.append(feats[j, :T])                        # (T, 64)
        return results

    def _pad(self, seq: np.ndarray) -> np.ndarray:
        T = len(seq)
        if T >= self.SEQ_LEN:
            return seq[:self.SEQ_LEN]
        pad = np.zeros((self.SEQ_LEN - T, seq.shape[1]), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)
