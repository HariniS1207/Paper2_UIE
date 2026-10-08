import torch
import torch.nn as nn


class ConsequenceEncoder(nn.Module):
    """Encode target-free enhancement state [X, Y, Y-X, alpha]."""

    def __init__(
        self,
        channels: int = 10,
        features: int = 32,
        embedding_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                channels,
                features,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features * 2,
                features * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Linear(
            features * 4,
            embedding_dim,
        )

    def encode_map(
        self,
        consequence: torch.Tensor,
    ):
        """
        Encode an already-computed consequence map.
        """

        features = self.encoder(consequence)

        pooled = self.pool(features).flatten(1)

        vector = self.projection(pooled)

        return vector

    def forward(self, original, enhanced, alpha):
        if original.ndim != 4 or enhanced.ndim != 4:
            raise ValueError("original and enhanced must have shape [B,C,H,W]")
        if original.shape != enhanced.shape or original.shape[1] != 3:
            raise ValueError("original and enhanced must be matching RGB tensors")
        if alpha.ndim == 0:
            alpha = alpha.reshape(1)
        if alpha.ndim != 1:
            raise ValueError("alpha must be a scalar tensor or have shape [B]")
        if alpha.numel() == 1:
            alpha = alpha.expand(original.shape[0])
        elif alpha.numel() != original.shape[0]:
            raise ValueError("alpha batch size must match the image batch")
        alpha = alpha.to(device=original.device, dtype=original.dtype)
        alpha_map = alpha[:, None, None, None].expand(-1, 1, *original.shape[-2:])
        consequence = torch.cat((original, enhanced, enhanced - original, alpha_map), dim=1)
        consequence_vector = self.encode_map(consequence)
        return consequence, consequence_vector
