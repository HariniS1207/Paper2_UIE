"""Capacity-scaled residual enhancer for the Paper 2 Phase 1 run.

This model preserves the original control equation and exact identity at
alpha=0. It is separate from ControllableEnhancer so prior checkpoints and
experiments retain their original architecture.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class FinalControllableEnhancer(nn.Module):
    """Residual image enhancer with an exact alpha-controlled identity path."""

    def __init__(self, channels: int = 3, features: int = 64, blocks: int = 4):
        super().__init__()
        if channels != 3:
            raise ValueError("FinalControllableEnhancer expects RGB input")
        if features <= 0 or blocks <= 0:
            raise ValueError("features and blocks must be positive")
        self.channels = channels
        self.features = features
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(features) for _ in range(blocks))
        )
        self.residual_head = nn.Conv2d(features, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected x with shape [B,3,H,W], got {tuple(x.shape)}")
        if alpha.ndim == 0:
            alpha = alpha.reshape(1)
        if alpha.ndim != 1:
            raise ValueError("alpha must have shape [B] or be a scalar tensor")
        if alpha.numel() == 1:
            alpha = alpha.expand(x.shape[0])
        elif alpha.numel() != x.shape[0]:
            raise ValueError("alpha batch size must match the image batch")
        if torch.any(alpha < 0) or torch.any(alpha > 1):
            raise ValueError("alpha must be in the range [0, 1]")
        alpha = alpha.to(device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        residual = self.residual_head(self.residual_blocks(self.stem(x)))
        return torch.clamp(x + alpha * residual, 0.0, 1.0)
