import torch

from src.losses.consequence import ConsequenceConsistencyLoss


def test_zero_loss_for_identical_representations():
    loss_fn = ConsequenceConsistencyLoss()

    representation = torch.rand(2, 128)

    loss = loss_fn(representation, representation)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_loss_is_nonnegative():
    loss_fn = ConsequenceConsistencyLoss()

    a = torch.rand(2, 128)
    b = torch.rand(2, 128)

    loss = loss_fn(a, b)

    assert loss.ndim == 0
    assert loss >= 0


def test_shape_mismatch_is_rejected():
    loss_fn = ConsequenceConsistencyLoss()

    a = torch.rand(2, 128)
    b = torch.rand(2, 64)

    try:
        loss_fn(a, b)
        assert False
    except ValueError:
        pass