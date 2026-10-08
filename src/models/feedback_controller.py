import torch
import torch.nn as nn


class FeedbackController(nn.Module):
    """
    Predicts an adjustment to the enhancement control parameter
    from the consequence representation.

    The predicted adjustment is bounded and applied to the
    current enhancement control value.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        delta_max: float = 0.10,
    ):
        super().__init__()

        self.controller = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_max = float(delta_max)

    def forward(
        self,
        consequence_vector: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:

        if consequence_vector.ndim != 2:
            raise ValueError(
                "consequence_vector must have shape [B, D]"
            )

        if alpha.ndim == 0:
            alpha = alpha.reshape(1)

        if alpha.ndim != 1:
            raise ValueError(
                "alpha must be a scalar tensor or shape [B]"
            )

        batch_size = consequence_vector.shape[0]

        if alpha.numel() == 1:
            alpha = alpha.expand(batch_size)
        elif alpha.numel() != batch_size:
            raise ValueError(
                f"alpha contains {alpha.numel()} values, "
                f"but batch size is {batch_size}"
            )

        alpha = alpha.to(
            device=consequence_vector.device,
            dtype=consequence_vector.dtype,
        )

        delta = self.delta_max * torch.tanh(
            self.controller(torch.cat((consequence_vector, alpha[:, None]), dim=1))
        ).squeeze(1)
        return delta


class AlphaOnlyController(nn.Module):
    """Alpha-only regression baseline, with the same bounded output range."""

    def __init__(self, hidden_dim: int = 32, delta_max: float = 0.10):
        super().__init__()
        self.controller = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_max = float(delta_max)

    def forward(self, alpha):
        if alpha.ndim == 0:
            alpha = alpha.reshape(1)
        if alpha.ndim != 1:
            raise ValueError("alpha must be scalar or have shape [B]")
        return self.delta_max * torch.tanh(self.controller(alpha[:, None])).squeeze(1)


class ImageStateController(nn.Module):
    """Image-state baseline using only X and Y_alpha, plus alpha."""

    def __init__(self, hidden_dim: int = 64, delta_max: float = 0.10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.controller = nn.Sequential(
            nn.Linear(65, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)
        )
        self.delta_max = float(delta_max)

    def forward(self, original, enhanced, alpha):
        if original.shape != enhanced.shape or original.ndim != 4 or original.shape[1] != 3:
            raise ValueError("original and enhanced must be matching RGB batches")
        if alpha.ndim == 0:
            alpha = alpha.reshape(1)
        if alpha.ndim != 1:
            raise ValueError("alpha must be scalar or have shape [B]")
        if alpha.numel() == 1:
            alpha = alpha.expand(original.shape[0])
        elif alpha.numel() != original.shape[0]:
            raise ValueError("alpha batch size must match the image batch")
        alpha = alpha.to(device=original.device, dtype=original.dtype)
        image_features = self.features(torch.cat((original, enhanced), dim=1))
        raw = self.controller(torch.cat((image_features, alpha[:, None]), dim=1))
        return self.delta_max * torch.tanh(raw).squeeze(1)
