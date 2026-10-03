import torch

from src.models.controllable_enhancer import ControllableEnhancer


def test_output_shape():
    model = ControllableEnhancer()

    x = torch.rand(2, 3, 256, 256)
    alpha = torch.tensor([0.0, 1.0])

    output = model(x, alpha)

    assert output.shape == x.shape


def test_output_range():
    model = ControllableEnhancer()

    x = torch.rand(2, 3, 256, 256)
    alpha = torch.tensor([0.0, 1.0])

    output = model(x, alpha)

    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)


def test_alpha_zero_preserves_input():
    model = ControllableEnhancer()

    x = torch.rand(1, 3, 64, 64)

    output = model(x, torch.tensor([0.0]))

    assert torch.allclose(output, x)


def test_alpha_changes_output():
    model = ControllableEnhancer()

    x = torch.rand(1, 3, 64, 64)

    conservative = model(x, torch.tensor([0.0]))
    aggressive = model(x, torch.tensor([1.0]))

    assert not torch.allclose(conservative, aggressive)