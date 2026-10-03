import torch
import torch.nn as nn


class ControllableEnhancer(nn.Module):
    """
    Minimal controllable enhancement network.

    The control parameter alpha determines the strength of the
    predicted residual enhancement.

    alpha = 0   -> no residual enhancement
    alpha = 1   -> full predicted residual
    """

    def __init__(self, channels: int = 3, features: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.residual_head = nn.Conv2d(
            features,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}"
            )

        if alpha.ndim == 0:
            alpha = alpha.reshape(1)

        if alpha.ndim != 1:
            raise ValueError(
                "alpha must have shape [B] or be a scalar tensor"
            )

        if alpha.numel() == 1:
            alpha = alpha.expand(x.shape[0])
        elif alpha.numel() != x.shape[0]:
            raise ValueError(
                f"alpha contains {alpha.numel()} values, "
                f"but batch size is {x.shape[0]}"
            )

        if torch.any(alpha < 0) or torch.any(alpha > 1):
            raise ValueError("alpha must be in the range [0, 1]")

        alpha = alpha.to(
            device=x.device,
            dtype=x.dtype,
        ).view(-1, 1, 1, 1)

        features = self.encoder(x)
        residual = self.residual_head(features)

        enhanced = x + alpha * residual

        return torch.clamp(enhanced, 0.0, 1.0)