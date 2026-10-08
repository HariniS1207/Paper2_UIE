import torch

from src.models.closed_loop import OneStepClosedLoop
from src.training.train_step import train_step


def test_train_step():
    model = OneStepClosedLoop()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    input_image = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    alpha = torch.tensor([0.25, 0.75])

    losses = train_step(
        model,
        optimizer,
        input_image,
        target,
        alpha,
    )

    assert set(losses) == {"loss", "identity_loss", "endpoint_loss"}
    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss >= 0
