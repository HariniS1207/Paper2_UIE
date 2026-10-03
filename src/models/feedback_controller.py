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
    ):
        super().__init__()

        self.controller = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

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

        adjustment = self.controller(consequence_vector).squeeze(1)

        updated_alpha = torch.clamp(
            alpha + adjustment,
            0.0,
            1.0,
        )

        return updated_alpha