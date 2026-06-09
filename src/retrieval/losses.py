"""Contrastive losses for pose-text retrieval.

Three components are implemented here:

1. ``info_nce`` - bidirectional CLIP-style contrastive loss between L2-normalized
   video and text embeddings. Optionally accepts a soft-positive mask so that
   semantically related captions (from the text semantic graph) are not treated
   as hard negatives.

2. ``density_loss`` - SignCL-style representation-density reduction
   (Ye et al., NeurIPS 2024). It operates on the per-frame memory of the pose
   encoder: frames close in time are pulled together, frames far apart in time
   are pushed away, which spreads out visually-similar-but-distinct signs.

3. ``retrieval_loss`` - convenience wrapper that combines the above.

All functions assume embeddings are float tensors and return scalar losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
    positive_mask: torch.Tensor | None = None,
    logit_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Bidirectional contrastive loss over a batch of paired embeddings.

    Args:
        video_embeddings: (B, D) tensor, expected L2-normalized.
        text_embeddings: (B, D) tensor, expected L2-normalized.
        temperature: softmax temperature (used only when ``logit_scale`` is None).
        positive_mask: optional (B, B) boolean/float matrix marking soft
            positives. The diagonal (exact pair) is always a positive. When
            given, each row's target distribution is uniform over its positives.
        logit_scale: optional multiplier on the cosine logits (CLIP-style
            learnable temperature). When given it overrides ``temperature`` and
            may be a tensor so gradients flow into a learnable parameter.

    Returns:
        Scalar loss = 0.5 * (video->text + text->video).
    """
    video_embeddings = F.normalize(video_embeddings.float(), dim=-1, eps=1e-6)
    text_embeddings = F.normalize(text_embeddings.float(), dim=-1, eps=1e-6)

    scale = logit_scale if logit_scale is not None else 1.0 / max(temperature, 1e-6)
    logits = (video_embeddings @ text_embeddings.T) * scale
    batch_size = logits.size(0)

    if positive_mask is None:
        targets = torch.arange(batch_size, device=logits.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    # Soft-positive targets: distribute probability mass over all positives.
    mask = positive_mask.to(dtype=logits.dtype, device=logits.device).clone()
    eye = torch.eye(batch_size, dtype=logits.dtype, device=logits.device)
    mask = torch.clamp(mask + eye, max=1.0)  # exact pair is always positive

    row_targets = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    col_targets = mask.T / mask.T.sum(dim=1, keepdim=True).clamp_min(1.0)

    v2t = -(row_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    t2v = -(col_targets * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (v2t + t2v)


def density_loss(
    memory: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 0.1,
    positive_window: int = 1,
    negative_margin: int = 8,
) -> torch.Tensor:
    """SignCL-style per-sequence frame contrastive loss.

    For every anchor frame, frames within ``positive_window`` time steps are
    positives and frames at least ``negative_margin`` steps away are negatives.
    Intermediate frames are ignored (they may belong to the same or a
    neighbouring sign and are ambiguous).

    Args:
        memory: (B, T, D) per-frame features from the pose encoder.
        valid_mask: (B, T) boolean mask, True for real frames.
        temperature: softmax temperature for the frame-level InfoNCE.
        positive_window: max |i - j| for a positive pair.
        negative_margin: min |i - j| for a negative pair.

    Returns:
        Scalar loss averaged over valid anchors that have at least one positive.
    """
    batch_size, seq_len, _ = memory.shape
    if seq_len <= positive_window + 1:
        return memory.new_zeros(())

    feats = F.normalize(memory.float(), dim=-1, eps=1e-6)

    # Time-distance based positive / negative masks, shared across the batch.
    idx = torch.arange(seq_len, device=memory.device)
    dist = (idx[None, :] - idx[:, None]).abs()  # (T, T)
    pos_time = (dist <= positive_window) & (dist > 0)
    neg_time = dist >= negative_margin

    total = memory.new_zeros(())
    count = 0
    for b in range(batch_size):
        valid = valid_mask[b]
        if valid.sum() <= positive_window + 1:
            continue

        sim = feats[b] @ feats[b].T / max(temperature, 1e-6)  # (T, T)
        valid_pair = valid[None, :] & valid[:, None]
        pos = pos_time & valid_pair
        neg = neg_time & valid_pair

        anchors = (pos.any(dim=1) & neg.any(dim=1)) & valid
        if not anchors.any():
            continue

        # For each anchor row: log-sum-exp over (positives + negatives) only.
        candidate = pos | neg
        neg_inf = torch.finfo(sim.dtype).min
        masked = sim.masked_fill(~candidate, neg_inf)
        log_denom = torch.logsumexp(masked, dim=1)  # (T,)

        pos_sim = sim.masked_fill(~pos, neg_inf)
        log_pos = torch.logsumexp(pos_sim, dim=1)  # (T,)

        row_loss = (log_denom - log_pos)[anchors]
        total = total + row_loss.mean()
        count += 1

    if count == 0:
        return memory.new_zeros(())
    return total / count


def retrieval_loss(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    temperature: float = 0.07,
    logit_scale: torch.Tensor | float | None = None,
    positive_mask: torch.Tensor | None = None,
    memory: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    density_weight: float = 0.0,
    density_temperature: float = 0.1,
    positive_window: int = 1,
    negative_margin: int = 8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine InfoNCE with the optional SignCL density term.

    Returns the total loss and a dict of scalar components for logging.
    """
    contrastive = info_nce(
        video_embeddings,
        text_embeddings,
        temperature=temperature,
        positive_mask=positive_mask,
        logit_scale=logit_scale,
    )
    parts = {"contrastive": float(contrastive.detach().cpu())}
    total = contrastive

    if density_weight > 0.0 and memory is not None and valid_mask is not None:
        density = density_loss(
            memory,
            valid_mask,
            temperature=density_temperature,
            positive_window=positive_window,
            negative_margin=negative_margin,
        )
        total = total + density_weight * density
        parts["density"] = float(density.detach().cpu())

    parts["total"] = float(total.detach().cpu())
    return total, parts
