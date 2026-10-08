import torch
import torch.nn as nn


class IdentityLoss(nn.Module):
    """
    Enforces identity behavior at alpha = 0.

    E(X, 0) should remain equal to the original input X.
    """

    def forward(self, output, input_image):
        if output.shape != input_image.shape:
            raise ValueError("output and input_image must have the same shape")

        return torch.mean(torch.abs(output - input_image))


class EndpointLoss(nn.Module):
    """
    Supervises the maximum-enhancement endpoint.

    At alpha = 1, the enhanced output is encouraged
    to approach the paired target.
    """

    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError("output and target must have the same shape")

        return torch.mean(torch.abs(output - target))
