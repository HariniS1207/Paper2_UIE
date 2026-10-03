import torch

from src.losses.control import ControlConsistencyLoss


def test_loss_is_zero_when_high_control_has_greater_change():
    loss_fn = ControlConsistencyLoss()

    x = torch.zeros(1, 3, 8, 8)
    low = torch.ones(1, 3, 8, 8) * 0.1
    high = torch.ones(1, 3, 8, 8) * 0.5

    loss = loss_fn(x, low, high)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_loss_is_positive_when_control_order_is_violated():
    loss_fn = ControlConsistencyLoss()

    x = torch.zeros(1, 3, 8, 8)
    low = torch.ones(1, 3, 8, 8) * 0.5
    high = torch.ones(1, 3, 8, 8) * 0.1

    loss = loss_fn(x, low, high)

    assert loss > 0


def test_loss_is_nonnegative():
    loss_fn = ControlConsistencyLoss()

    x = torch.rand(2, 3, 16, 16)
    low = torch.rand(2, 3, 16, 16)
    high = torch.rand(2, 3, 16, 16)

    loss = loss_fn(x, low, high)

    assert loss.ndim == 0
    assert loss >= 0
    