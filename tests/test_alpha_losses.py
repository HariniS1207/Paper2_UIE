import torch

from src.losses.alpha import IdentityLoss, EndpointLoss


def test_identity_loss_zero_for_identical_images():
    image = torch.rand(2, 3, 16, 16)

    loss_fn = IdentityLoss()
    loss = loss_fn(image, image)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_identity_loss_positive_for_different_images():
    input_image = torch.zeros(2, 3, 16, 16)
    output = torch.ones(2, 3, 16, 16)

    loss_fn = IdentityLoss()
    loss = loss_fn(output, input_image)

    assert loss > 0


def test_endpoint_loss_zero_for_identical_target():
    target = torch.rand(2, 3, 16, 16)

    loss_fn = EndpointLoss()
    loss = loss_fn(target, target)

    assert torch.isclose(loss, torch.tensor(0.0))


def test_endpoint_loss_positive_for_different_output():
    output = torch.zeros(2, 3, 16, 16)
    target = torch.ones(2, 3, 16, 16)

    loss_fn = EndpointLoss()
    loss = loss_fn(output, target)

    assert loss > 0