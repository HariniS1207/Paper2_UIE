import torch

from src.losses.reconstruction import ReconstructionLoss


def test_zero_loss_for_identical_images():
    loss_fn = ReconstructionLoss()

    image = torch.rand(2, 3, 32, 32)

    loss = loss_fn(image, image)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_loss_is_nonnegative():
    loss_fn = ReconstructionLoss()

    prediction = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)

    loss = loss_fn(prediction, target)

    assert loss.ndim == 0
    assert loss >= 0