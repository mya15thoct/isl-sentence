"""
Seq2Seq with Cross-Attention for ISL Sentence Recognition.

Instead of CTC (needs model to discover alignment from sequence-level labels),
this uses cross-attention + teacher forcing for direct per-step supervision.

Architecture:
  Encoder: MLP branches + BiLSTM (transfer from word model)
  Decoder: Embedding + LSTM + Bahdanau Attention → predict glosses step-by-step

Special tokens:
  SOS = num_glosses      (start of sequence)
  EOS = num_glosses + 1  (end of sequence)

Usage:
  model = Seq2SeqModel(num_glosses=80)
  model.transfer_encoder_weights('/path/to/best_model_combined')
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHECKPOINT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# MLP BRANCHES (same as sentence_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def _pose_branch(name='pose'):
    return tf.keras.Sequential([
        layers.Dense(128, activation='relu'), layers.BatchNormalization(),
        layers.Dense(64,  activation='relu'), layers.BatchNormalization(),
    ], name=name)

def _face_branch(name='face'):
    return tf.keras.Sequential([
        layers.Dense(512, activation='relu'), layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(),
        layers.Dense(128, activation='relu'), layers.BatchNormalization(),
        layers.Dense(128, activation='relu'), layers.BatchNormalization(),
    ], name=name)

def _hand_branch(name='hand'):
    return tf.keras.Sequential([
        layers.Dense(256, activation='relu'), layers.BatchNormalization(),
        layers.Dense(128, activation='relu'), layers.BatchNormalization(),
        layers.Dense(64,  activation='relu'), layers.BatchNormalization(),
    ], name=name)


# ─────────────────────────────────────────────────────────────────────────────
# BAHDANAU ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W_enc = layers.Dense(units)
        self.W_dec = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, decoder_state, encoder_outputs):
        """
        Args:
            decoder_state:   (B, dec_units)
            encoder_outputs: (B, T, enc_units)
        Returns:
            context: (B, enc_units),  weights: (B, T, 1)
        """
        query = tf.expand_dims(decoder_state, 1)
        score = self.V(tf.nn.tanh(self.W_enc(encoder_outputs) + self.W_dec(query)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * encoder_outputs, axis=1)
        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
# SEQ2SEQ MODEL
# ─────────────────────────────────────────────────────────────────────────────

class Seq2SeqModel(Model):
    """
    Encoder-Decoder with attention for continuous sign language recognition.

    Encoder processes (B, T, 1662) keypoints → (B, T, enc_dim) hidden states.
    Decoder autoregressively predicts gloss sequence with cross-attention.
    """

    def __init__(self, num_glosses, enc_units=32, dec_units=64,
                 embed_dim=64, max_decode_len=15, **kwargs):
        super().__init__(**kwargs)

        self.num_glosses = num_glosses
        self.vocab_size = num_glosses + 2   # +SOS +EOS
        self.sos_id = num_glosses
        self.eos_id = num_glosses + 1
        self.dec_units = dec_units
        self.enc_dim = enc_units * 2        # BiLSTM output dim
        self.max_decode_len = max_decode_len

        # ── Encoder ──
        self.enc_masking = layers.Masking(mask_value=0.0)
        self.pose_br = _pose_branch('pose')
        self.face_br = _face_branch('face')
        self.hand_br = _hand_branch('hand')
        self.shared1 = layers.Dense(256, activation='relu', name='shared1')
        self.shared2 = layers.Dense(128, activation='relu', name='shared2')
        self.enc_drop = layers.Dropout(0.3)
        self.bilstm1 = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True), name='bilstm1')
        self.bilstm2 = layers.Bidirectional(
            layers.LSTM(enc_units, return_sequences=True, return_state=True),
            name='bilstm2')

        # Project BiLSTM final state → decoder initial state
        self.state_h_proj = layers.Dense(dec_units, activation='tanh', name='state_h_proj')
        self.state_c_proj = layers.Dense(dec_units, activation='tanh', name='state_c_proj')

        # ── Decoder ──
        self.dec_embedding = layers.Embedding(self.vocab_size, embed_dim, name='dec_embed')
        self.attention = BahdanauAttention(dec_units, name='attention')
        self.decoder_lstm = layers.LSTMCell(dec_units, name='dec_lstm_cell')
        self.dec_dropout = layers.Dropout(0.3)
        self.output_proj = layers.Dense(self.vocab_size, name='output_proj')

    def encode(self, x, training=False):
        """Encode keypoints → encoder_outputs, (state_h, state_c)."""
        x = self.enc_masking(x)

        # Split + MLP (TimeDistributed via reshape)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        pose = tf.reshape(x[:, :, :132],    (B * T, 132))
        face = tf.reshape(x[:, :, 132:1536],(B * T, 1404))
        hand = tf.reshape(x[:, :, 1536:],   (B * T, 126))

        pose_f = tf.reshape(self.pose_br(pose, training=training), (B, T, 64))
        face_f = tf.reshape(self.face_br(face, training=training), (B, T, 128))
        hand_f = tf.reshape(self.hand_br(hand, training=training), (B, T, 64))

        merged = tf.concat([pose_f, face_f, hand_f], axis=-1)  # (B,T,256)

        # Shared Dense (TimeDistributed via reshape)
        flat = tf.reshape(merged, (B * T, 256))
        flat = self.shared1(flat)
        flat = self.shared2(flat)
        x = tf.reshape(flat, (B, T, 128))
        x = self.enc_drop(x, training=training)

        # BiLSTM
        x = self.bilstm1(x, training=training)
        x = self.enc_drop(x, training=training)

        out = self.bilstm2(x, training=training)
        enc_outputs = out[0]                              # (B, T, enc_dim)
        fwd_h, fwd_c, bwd_h, bwd_c = out[1], out[2], out[3], out[4]

        state_h = self.state_h_proj(tf.concat([fwd_h, bwd_h], -1))
        state_c = self.state_c_proj(tf.concat([fwd_c, bwd_c], -1))

        return enc_outputs, (state_h, state_c)

    def decode_step(self, token_id, state, enc_outputs, training=False):
        """One decoder step: token → logits, new_state, attention_weights."""
        embed = self.dec_embedding(token_id)             # (B, embed_dim)
        context, att_w = self.attention(state[0], enc_outputs)
        lstm_input = tf.concat([embed, context], axis=-1)
        output, new_state = self.decoder_lstm(lstm_input, state)
        output = self.dec_dropout(output, training=training)
        logits = self.output_proj(output)                # (B, vocab_size)
        return logits, new_state, att_w

    def call(self, inputs, training=False):
        """
        Forward pass with teacher forcing.

        Args:
            inputs: tuple (encoder_input, decoder_input)
              encoder_input: (B, T, 1662)
              decoder_input: (B, L) — [SOS, g1, g2, ...] (shifted right)

        Returns:
            logits: (B, L, vocab_size)
        """
        enc_input, dec_input = inputs
        enc_outputs, state = self.encode(enc_input, training=training)

        B = tf.shape(dec_input)[0]
        L = tf.shape(dec_input)[1]

        # Collect logits for each decoder step
        all_logits = tf.TensorArray(dtype=tf.float32, size=L)

        for t in tf.range(L):
            token = dec_input[:, t]
            logits, state, _ = self.decode_step(token, state, enc_outputs, training)
            all_logits = all_logits.write(t, logits)

        return tf.transpose(all_logits.stack(), [1, 0, 2])  # (B, L, vocab)

    def greedy_decode(self, encoder_input):
        """Inference: greedy autoregressive decode."""
        enc_outputs, state = self.encode(encoder_input, training=False)
        B = tf.shape(encoder_input)[0]

        token = tf.fill([B], self.sos_id)
        result = []

        for _ in range(self.max_decode_len):
            logits, state, _ = self.decode_step(token, state, enc_outputs)
            token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            result.append(token.numpy())
            if tf.reduce_all(token == self.eos_id):
                break

        # (max_steps, B) → (B, max_steps), trim EOS
        preds = np.array(result).T
        decoded = []
        for seq in preds:
            trimmed = []
            for t in seq:
                if t == self.eos_id:
                    break
                if t != self.sos_id:
                    trimmed.append(int(t))
            decoded.append(trimmed)
        return decoded

    def transfer_encoder_weights(self, word_model_path: str):
        """Transfer MLP branch + BiLSTM weights from word model."""
        print(f'[Seq2Seq] Loading word model from: {word_model_path}')
        try:
            word_model = tf.keras.models.load_model(str(word_model_path))
        except Exception as e:
            print(f'[Seq2Seq] Could not load word model: {e}')
            return

        transferred, skipped = 0, 0
        for layer in self.layers:
            try:
                src = word_model.get_layer(layer.name)
                w = src.get_weights()
                if w:
                    layer.set_weights(w)
                    transferred += 1
            except Exception:
                skipped += 1
        print(f'[Seq2Seq] Transferred: {transferred}  |  Skipped: {skipped}')
