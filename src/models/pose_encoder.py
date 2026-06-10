"""
Holistic keypoint Conformer encoder for SignPose2Text experiments.

Input:
  x       : (B, T, 1662) MediaPipe Holistic keypoints
  lengths : (B,) valid frame counts before padding

Output:
  embedding : (B, 384), L2-normalized by default

Architecture:
  body-part MLP branches -> temporal downsampling -> Conformer blocks
  -> attention pooling -> projection.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


POSE_DIM = 132
FACE_DIM = 1404
HANDS_DIM = 126
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HANDS_DIM


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return True for valid positions, False for padding."""
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def downsample_lengths(lengths: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    """Conv1d output lengths for each sequence."""
    return torch.div(lengths + 2 * padding - kernel_size, stride, rounding_mode="floor") + 1


class MLPBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BodyPartFrameEncoder(nn.Module):
    """Per-frame pose/face/hands encoder."""

    def __init__(
        self,
        pose_out: int = 64,
        face_out: int = 128,
        hands_out: int = 128,
        model_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pose_branch = MLPBranch(POSE_DIM, 128, pose_out, dropout)
        self.face_branch = MLPBranch(FACE_DIM, 512, face_out, dropout)
        self.hands_branch = MLPBranch(HANDS_DIM, 256, hands_out, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(pose_out + face_out + hands_out, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != KEYPOINT_DIM:
            raise ValueError(f"Expected last dim {KEYPOINT_DIM}, got {x.size(-1)}")

        pose = x[..., :POSE_DIM]
        face = x[..., POSE_DIM : POSE_DIM + FACE_DIM]
        hands = x[..., POSE_DIM + FACE_DIM :]

        features = torch.cat(
            [
                self.pose_branch(pose),
                self.face_branch(face),
                self.hands_branch(hands),
            ],
            dim=-1,
        )
        return self.fusion(features)


class HandAwareFrameEncoder(nn.Module):
    """Hand-centric per-frame encoder.

    The two hands form the main path that always flows through. Pose and face
    only *add* context via residual cross-attention (query = hands), so the
    hand signal can never be down-weighted away - it is the backbone, the rest
    are modulators.
    """

    HAND_DIM = HANDS_DIM // 2  # 63 per hand

    def __init__(self, model_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.left_branch = MLPBranch(self.HAND_DIM, 128, model_dim, dropout)
        self.right_branch = MLPBranch(self.HAND_DIM, 128, model_dim, dropout)
        self.pose_branch = MLPBranch(POSE_DIM, 128, model_dim, dropout)
        self.face_branch = MLPBranch(FACE_DIM, 256, model_dim, dropout)

        self.hand_fuse = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
        )
        # Lightweight cross-attention (hands query 2 context tokens) done with
        # plain matmul to avoid the huge B*T batch that breaks the SDPA kernel.
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.attn_scale = model_dim ** -0.5
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(model_dim)
        self.out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != KEYPOINT_DIM:
            raise ValueError(f"Expected last dim {KEYPOINT_DIM}, got {x.size(-1)}")

        pose = x[..., :POSE_DIM]
        face = x[..., POSE_DIM : POSE_DIM + FACE_DIM]
        hands_start = POSE_DIM + FACE_DIM
        left = x[..., hands_start : hands_start + self.HAND_DIM]
        right = x[..., hands_start + self.HAND_DIM :]

        left_feat = self.left_branch(left)
        right_feat = self.right_branch(right)
        pose_feat = self.pose_branch(pose)
        face_feat = self.face_branch(face)

        # Main path: both hands fused.
        hands = self.hand_fuse(torch.cat([left_feat, right_feat], dim=-1))  # (B, T, D)

        # Context (pose + face) added via residual cross-attention queried by
        # hands. Manual single-head attention over the 2 context tokens keeps
        # everything in (B, T, ...) shape - no B*T flattening.
        context = torch.stack([pose_feat, face_feat], dim=2)  # (B, T, 2, D)
        q = self.q_proj(hands)                                # (B, T, D)
        k = self.k_proj(context)                              # (B, T, 2, D)
        v = self.v_proj(context)                              # (B, T, 2, D)
        scores = torch.einsum("btd,btnd->btn", q, k) * self.attn_scale  # (B, T, 2)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))      # (B, T, 2)
        attended = torch.einsum("btn,btnd->btd", weights, v)           # (B, T, D)
        fused = self.attn_norm(hands + attended)
        return self.out(fused)


class TemporalDownsample(nn.Module):
    def __init__(self, model_dim: int = 256, stride: int = 4, kernel_size: int = 7):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B,T,D) -> (B,D,T) -> (B,T',D)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        x = self.norm(x)
        lengths = downsample_lengths(lengths, self.kernel_size, self.stride, self.padding)
        lengths = torch.clamp(lengths, min=1, max=x.size(1))
        return x, lengths


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = model_dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, model_dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        norm_groups = min(8, model_dim)
        while model_dim % norm_groups != 0:
            norm_groups -= 1

        self.norm = nn.LayerNorm(model_dim)
        self.pointwise_in = nn.Conv1d(model_dim, model_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=model_dim,
        )
        # Keep the attribute name for checkpoint compatibility, but avoid
        # BatchNorm running stats because long AMP runs can corrupt them.
        self.batch_norm = nn.GroupNorm(num_groups=norm_groups, num_channels=model_dim)
        self.pointwise_out = nn.Conv1d(model_dim, model_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_in(x)
        x = F.glu(x, dim=1)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int = 256,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForward(model_dim, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(model_dim, conv_kernel, dropout)
        self.ff2 = FeedForward(model_dim, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~valid_mask

        x = x + 0.5 * self.ff1(x)

        attn_input = self.attn_norm(x)
        attn_out, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)

        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.final_norm(x)

        return x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class AttentionPool(nn.Module):
    def __init__(self, model_dim: int = 256):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid_mask, mask_value)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * weights, dim=1)


class KeypointConformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 256,
        projection_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 4,
        downsample_stride: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        normalize_output: bool = True,
        hand_aware: bool = False,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        if hand_aware:
            self.frame_encoder = HandAwareFrameEncoder(
                model_dim=model_dim, num_heads=num_heads, dropout=dropout
            )
        else:
            self.frame_encoder = BodyPartFrameEncoder(model_dim=model_dim, dropout=dropout)
        self.downsample = TemporalDownsample(model_dim=model_dim, stride=downsample_stride)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pool = AttentionPool(model_dim)
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, projection_dim),
        )

    def encode_sequence(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return temporal memory, valid mask, and downsampled lengths."""
        if lengths is None:
            lengths = torch.full(
                (x.size(0),),
                x.size(1),
                device=x.device,
                dtype=torch.long,
            )
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long)

        x = self.frame_encoder(x)
        x, lengths = self.downsample(x, lengths)
        valid_mask = lengths_to_mask(lengths, x.size(1))

        for layer in self.layers:
            x = layer(x, valid_mask)

        return x, valid_mask, lengths

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x, valid_mask, _ = self.encode_sequence(x, lengths)
        pooled = self.pool(x, valid_mask)
        embedding = self.projection(pooled)
        if self.normalize_output:
            embedding = F.normalize(embedding.float(), dim=-1, eps=1e-6)
        return embedding


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = KeypointConformerEncoder()
    x = torch.randn(2, 512, 1662)
    lengths = torch.tensor([512, 317])
    y = model(x, lengths)
    print("output:", tuple(y.shape))
    print("params:", count_parameters(model))
