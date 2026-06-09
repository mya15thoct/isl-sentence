"""Trainable sentence-transformer text encoder for pose-text retrieval.

Unlike the earlier pipeline (which froze MiniLM and cached fixed embeddings),
this encoder is fine-tuned end-to-end together with the pose encoder. The text
space is therefore free to reshape so that ISL captions become more separable -
the single biggest lever for retrieval quality (CLIP / SignCLIP style).
"""

from __future__ import annotations

import importlib.util

import torch
from torch import nn


DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _require_transformers() -> None:
    if importlib.util.find_spec("transformers") is None:
        raise ImportError(
            "transformers is required for the text encoder. "
            "Install it in the training environment, e.g. `pip install transformers`."
        )


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean of token embeddings weighted by the attention mask."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-9)
    return summed / counts


class TextEncoder(nn.Module):
    """MiniLM/Sentence-BERT encoder with mean pooling and optional projection."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        output_dim: int = 384,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        _require_transformers()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

        hidden = int(self.encoder.config.hidden_size)
        self.output_dim = output_dim
        self.projection = nn.Linear(hidden, output_dim) if hidden != output_dim else None

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        if self.projection is not None:
            pooled = self.projection(pooled)
        return pooled

    @torch.no_grad()
    def encode_texts(self, texts: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
        """Convenience helper to embed raw strings (eval / indexing)."""
        batch = self.tokenize(texts)
        batch = {key: value.to(device) for key, value in batch.items()}
        return self.forward(batch["input_ids"], batch["attention_mask"])
