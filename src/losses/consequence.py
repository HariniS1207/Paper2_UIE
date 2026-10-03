import torch
import torch.nn as nn


class ConsequenceConsistencyLoss(nn.Module):
    """
    Measures consistency between two consequence representations.

    Used to encourage the consequence representation to remain
    stable when the same enhancement process is evaluated through
    different reconstruction paths.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(
        self,
        consequence_a: torch.Tensor,
        consequence_b: torch.Tensor,
    ) -> torch.Tensor:

        if consequence_a.shape != consequence_b.shape:
            raise ValueError(
                "consequence representations must have identical shapes"
            )

        return self.loss(consequence_a, consequence_b)