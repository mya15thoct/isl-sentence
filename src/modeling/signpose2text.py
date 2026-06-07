"""Conformer-to-T5 SignPose2Text model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.modeling.keypoint_conformer import KeypointConformerEncoder


def require_transformers() -> None:
    if importlib.util.find_spec("transformers") is None:
        raise ImportError(
            "transformers is required for SignPose2Text. "
            "Install it in the training environment, e.g. `pip install transformers`."
        )


class ConformerT5SignPose2Text(nn.Module):
    """Use a holistic Conformer as the encoder memory for a T5 decoder."""

    def __init__(
        self,
        text_model_name: str = "t5-small",
        keypoint_model_dim: int = 256,
        keypoint_layers: int = 4,
        keypoint_heads: int = 4,
        downsample_stride: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        require_transformers()
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_name)
        self.keypoint_encoder = KeypointConformerEncoder(
            model_dim=keypoint_model_dim,
            projection_dim=keypoint_model_dim,
            num_layers=keypoint_layers,
            num_heads=keypoint_heads,
            downsample_stride=downsample_stride,
            dropout=dropout,
            normalize_output=False,
        )
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(keypoint_model_dim),
            nn.Linear(keypoint_model_dim, self.text_model.config.d_model),
        )

    def encode_keypoints(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, valid_mask, _ = self.keypoint_encoder.encode_sequence(keypoints, lengths)
        memory = self.memory_projection(memory)
        return memory, valid_mask.long()

    def forward(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> Any:
        from transformers.modeling_outputs import BaseModelOutput

        memory, attention_mask = self.encode_keypoints(keypoints, lengths)
        encoder_outputs = BaseModelOutput(last_hidden_state=memory)
        return self.text_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        keypoints: torch.Tensor,
        lengths: torch.Tensor,
        max_new_tokens: int = 64,
        num_beams: int = 4,
    ) -> torch.Tensor:
        from transformers.modeling_outputs import BaseModelOutput

        memory, attention_mask = self.encode_keypoints(keypoints, lengths)
        encoder_outputs = BaseModelOutput(last_hidden_state=memory)
        return self.text_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    def decode(self, token_ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def load_tokenizer(text_model_name: str):
    require_transformers()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(text_model_name)
