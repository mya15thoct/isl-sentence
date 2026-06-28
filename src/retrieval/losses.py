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


def _masked_soft_ce(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy with soft-positive targets and ignored columns.

    ``pos_mask`` (Q, C) marks positives (each row normalized to a target
    distribution). ``ignore_mask`` (Q, C) marks columns removed from the
    denominator entirely - neither positive nor negative - which is how
    redundant same-caption queue entries are kept from acting as false
    negatives.
    """
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(ignore_mask, neg_inf)
    log_prob = torch.log_softmax(logits, dim=1)
    target = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(target * log_prob).sum(dim=1).mean()


def info_nce_xbm(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    queue_video: torch.Tensor | None = None,
    queue_text: torch.Tensor | None = None,
    queue_groups: torch.Tensor | None = None,
    in_batch_positive: torch.Tensor | None = None,
    temperature: float = 0.07,
    logit_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Bidirectional InfoNCE with a cross-batch memory bank of negatives.

    The in-batch text/video embeddings are augmented with ``queue_*`` embeddings
    from previous batches, giving thousands of extra negatives at almost no GPU
    cost (the queue is detached). Queue entries that share a caption group with
    the query are *ignored* (not treated as hard negatives), preserving the
    redundancy-aware objective across batches.

    ``in_batch_positive`` is the off-diagonal (B, B) soft-positive mask
    (redundancy group + semantic neighbours); the exact pair (diagonal) is always
    a positive and is added here.
    """
    v = F.normalize(video_embeddings.float(), dim=-1, eps=1e-6)
    t = F.normalize(text_embeddings.float(), dim=-1, eps=1e-6)
    batch_size = v.size(0)
    device = v.device
    group_ids = group_ids.to(device)
    scale = logit_scale if logit_scale is not None else 1.0 / max(temperature, 1e-6)

    def _with_queue(in_batch: torch.Tensor, queue: torch.Tensor | None) -> torch.Tensor:
        if queue is not None and queue.numel():
            return torch.cat([in_batch, queue.to(device=device, dtype=in_batch.dtype)], dim=0)
        return in_batch

    def _col_groups() -> torch.Tensor:
        if queue_groups is not None and queue_groups.numel():
            return torch.cat([group_ids, queue_groups.to(device)], dim=0)
        return group_ids

    def _masks(col_groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cols = col_groups.size(0)
        pos = torch.zeros(batch_size, cols, device=device)
        diag = torch.arange(batch_size, device=device)
        pos[diag, diag] = 1.0  # exact pair is always a positive
        if in_batch_positive is not None:
            ib = in_batch_positive.to(device=device, dtype=pos.dtype)
            pos[:, :batch_size] = torch.maximum(pos[:, :batch_size], ib)
        same = group_ids.view(-1, 1) == col_groups.view(1, -1)
        ignore = same & (pos == 0)  # same-caption non-positives -> drop, not penalize
        return pos, ignore

    col_groups = _col_groups()
    pos_mask, ignore_mask = _masks(col_groups)

    # video -> text (candidates = in-batch texts + queued texts)
    logits_v2t = (v @ _with_queue(t, queue_text).T) * scale
    loss_v2t = _masked_soft_ce(logits_v2t, pos_mask, ignore_mask)

    # text -> video (candidates = in-batch videos + queued videos)
    logits_t2v = (t @ _with_queue(v, queue_video).T) * scale
    loss_t2v = _masked_soft_ce(logits_t2v, pos_mask, ignore_mask)

    return 0.5 * (loss_v2t + loss_t2v)


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


if __name__ == "__main__":  # smoke test: `python -m src.retrieval.losses`
    torch.manual_seed(0)
    B, D, Q = 8, 16, 40
    v = torch.randn(B, D)
    t = torch.randn(B, D)
    g = torch.arange(B)  # all distinct groups

    # With an empty queue and no in-batch positives, the XBM loss must match the
    # plain bidirectional InfoNCE (sanity check the masking/targets).
    a = info_nce(v, t)
    b = info_nce_xbm(v, t, g)
    assert torch.allclose(a, b, atol=1e-5), (a.item(), b.item())

    # With a populated queue the loss is finite and the queue adds negatives.
    qv = F.normalize(torch.randn(Q, D), dim=-1)
    qt = F.normalize(torch.randn(Q, D), dim=-1)
    qg = torch.randint(100, 200, (Q,))  # unrelated groups
    c = info_nce_xbm(v, t, g, queue_video=qv, queue_text=qt, queue_groups=qg)
    assert torch.isfinite(c)

    # A queued entry sharing a query's group must be ignored, not a hard negative.
    qg2 = qg.clone()
    qg2[0] = g[0]  # queue[0] now shares group with query 0
    d = info_nce_xbm(v, t, g, queue_video=qv, queue_text=qt, queue_groups=qg2)
    assert torch.isfinite(d)
    print(f"ok: info_nce={a.item():.4f} xbm_empty={b.item():.4f} "
          f"xbm_queue={c.item():.4f} xbm_redundant={d.item():.4f}")
