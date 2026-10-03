import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    Basic reconstruction loss for enhanced output
    against a target reference.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have identical shapes"
            )

        return self.loss(prediction, target)