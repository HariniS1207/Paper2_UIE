import torch
import torch.nn as nn


class ReDegradation(nn.Module):
    """
    Differentiable re-degradation operator.

    Reconstructs a degraded observation from an enhanced image.
    This module is used to generate consequences of enhancement
    for the later closed-loop feedback stage.
    """

    def __init__(self, channels: int = 3):
        super().__init__()

        self.operator = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        if enhanced.ndim != 4:
            raise ValueError(
                "Expected enhanced image with shape [B,C,H,W], "
                f"got {tuple(enhanced.shape)}"
            )

        return self.operator(enhanced)