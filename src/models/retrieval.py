"""Dual-encoder model for ISL pose-text retrieval.

A pose (Keypoint Conformer) encoder and a text (MiniLM) encoder project pose
sequences and captions into a shared embedding space. Embeddings are returned
un-normalized; loss functions and evaluation normalize them before computing
cosine similarity. The pose encoder also exposes its per-frame memory so the
SignCL density loss can operate on it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.pose_encoder import KeypointConformerEncoder
from src.models.text_encoder import DEFAULT_TEXT_MODEL, TextEncoder


class PoseTextRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 384,
        pose_model_dim: int = 256,
        pose_layers: int = 4,
        pose_heads: int = 4,
        downsample_stride: int = 4,
        dropout: float = 0.1,
        hand_aware: bool = False,
        context_parts: tuple[str, ...] = ("pose", "face"),
        motion: bool = False,
        pose_pooling: str = "attention",
        frame_encoder: str = "parts",
        temporal: str = "conformer",
        text_model_name: str = DEFAULT_TEXT_MODEL,
        max_text_length: int = 64,
        text_pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.pose_encoder = KeypointConformerEncoder(
            model_dim=pose_model_dim,
            projection_dim=embedding_dim,
            num_layers=pose_layers,
            num_heads=pose_heads,
            downsample_stride=downsample_stride,
            dropout=dropout,
            normalize_output=False,
            hand_aware=hand_aware,
            context_parts=context_parts,
            motion=motion,
            pool_type=pose_pooling,
            frame_encoder=frame_encoder,
            temporal=temporal,
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            output_dim=embedding_dim,
            max_length=max_text_length,
            pooling=text_pooling,
        )
        self.embedding_dim = embedding_dim
        # CLIP-style learnable temperature (stored as log scale).
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def current_logit_scale(self, max_scale: float = 100.0) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=max_scale)

    @property
    def tokenizer(self):
        return self.text_encoder.tokenizer

    def encode_pose(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (video embedding, per-frame memory, valid frame mask)."""
        memory, valid_mask, _ = self.pose_encoder.encode_sequence(keypoints, lengths)
        pooled = self.pose_encoder.pool(memory, valid_mask)
        embedding = self.pose_encoder.projection(pooled)
        return embedding, memory, valid_mask

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        video_embedding, memory, valid_mask = self.encode_pose(keypoints, lengths)
        text_embedding = self.encode_text(input_ids, attention_mask)
        return {
            "video_embedding": video_embedding,
            "text_embedding": text_embedding,
            "memory": memory,
            "valid_mask": valid_mask,
        }
