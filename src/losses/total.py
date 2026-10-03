import torch
import torch.nn as nn

from src.losses.reconstruction import ReconstructionLoss
from src.losses.consequence import ConsequenceConsistencyLoss
from src.losses.control import ControlConsistencyLoss


class TotalLoss(nn.Module):
    """
    Weighted combination of the Paper 2 training objectives.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        consequence_weight: float = 0.1,
        control_weight: float = 0.1,
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.consequence_weight = consequence_weight
        self.control_weight = control_weight

        self.reconstruction_loss = ReconstructionLoss()
        self.consequence_loss = ConsequenceConsistencyLoss()
        self.control_loss = ControlConsistencyLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        consequence_a: torch.Tensor,
        consequence_b: torch.Tensor,
        input_image: torch.Tensor,
        low_output: torch.Tensor,
        high_output: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction = self.reconstruction_loss(
            prediction,
            target,
        )

        consequence = self.consequence_loss(
            consequence_a,
            consequence_b,
        )

        control = self.control_loss(
            input_image,
            low_output,
            high_output,
        )

        total = (
            self.reconstruction_weight * reconstruction
            + self.consequence_weight * consequence
            + self.control_weight * control
        )

        return total