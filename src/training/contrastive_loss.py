from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SymmetricContrastiveLoss(nn.Module):
    """CLIP-style video-text contrastive loss."""

    def forward(
        self,
        video_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        video_embeddings = torch.nan_to_num(
            video_embeddings.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        text_embeddings = torch.nan_to_num(
            text_embeddings.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        video_embeddings = F.normalize(video_embeddings, dim=-1, eps=1e-6)
        text_embeddings = F.normalize(text_embeddings, dim=-1, eps=1e-6)

        scale = logit_scale.float().exp().clamp(max=100.0)
        logits = scale * video_embeddings @ text_embeddings.t()
        labels = torch.arange(logits.size(0), device=logits.device)

        video_to_text = F.cross_entropy(logits, labels)
        text_to_video = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (video_to_text + text_to_video)

        with torch.no_grad():
            v2t_acc = (logits.argmax(dim=1) == labels).float().mean()
            t2v_acc = (logits.argmax(dim=0) == labels).float().mean()

        metrics = {
            "loss": float(loss.detach().cpu()),
            "v2t_acc": float(v2t_acc.detach().cpu()),
            "t2v_acc": float(t2v_acc.detach().cpu()),
            "logit_scale": float(logit_scale.exp().detach().cpu()),
        }
        return loss, metrics
