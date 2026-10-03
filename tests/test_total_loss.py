import torch

from src.losses.total import TotalLoss


def test_total_loss_returns_scalar():
    loss_fn = TotalLoss()

    prediction = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 16, 16)

    consequence_a = torch.rand(2, 128)
    consequence_b = torch.rand(2, 128)

    x = torch.rand(2, 3, 16, 16)
    low = torch.rand(2, 3, 16, 16)
    high = torch.rand(2, 3, 16, 16)

    loss = loss_fn(
        prediction,
        target,
        consequence_a,
        consequence_b,
        x,
        low,
        high,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_zero_components_produce_zero_loss():
    loss_fn = TotalLoss()

    x = torch.rand(1, 3, 8, 8)

    loss = loss_fn(
        x,
        x,
        torch.zeros(1, 128),
        torch.zeros(1, 128),
        x,
        x,
        x,
    )

    assert torch.allclose(loss, torch.tensor(0.0))