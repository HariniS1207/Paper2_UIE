import torch
import torch.nn as nn


class ControlConsistencyLoss(nn.Module):
    """
    Encourages the enhancement response to follow the requested
    control strength.

    For alpha_low < alpha_high, the higher-alpha output should
    remain at least as different from the input as the lower-alpha
    output.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_image: torch.Tensor,
        low_output: torch.Tensor,
        high_output: torch.Tensor,
    ) -> torch.Tensor:

        if (
            input_image.shape != low_output.shape
            or input_image.shape != high_output.shape
        ):
            raise ValueError(
                "All images must have identical shapes"
            )

        low_change = torch.mean(
            torch.abs(low_output - input_image),
            dim=(1, 2, 3),
        )

        high_change = torch.mean(
            torch.abs(high_output - input_image),
            dim=(1, 2, 3),
        )

        return torch.relu(low_change - high_change).mean()