import torch
import torch.nn as nn


class ConsequenceEncoder(nn.Module):
    """
    Encodes the consequence of enhancement.

    The consequence is represented using the difference between
    the original underwater image and its re-degraded reconstruction.

    Output:
        consequence_map: spatial representation
        consequence_vector: compact global representation
    """

    def __init__(
        self,
        channels: int = 3,
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

    def forward(
        self,
        original: torch.Tensor,
        redegraded: torch.Tensor,
    ):
        if original.ndim != 4 or redegraded.ndim != 4:
            raise ValueError(
                "Both inputs must have shape [B,C,H,W]"
            )

        if original.shape != redegraded.shape:
            raise ValueError(
                "original and redegraded must have identical shapes"
            )

        consequence = torch.abs(original - redegraded)

        features = self.encoder(consequence)

        pooled = self.pool(features).flatten(1)

        consequence_vector = self.projection(pooled)

        return consequence, consequence_vector