import torch

from src.models.closed_loop import OneStepClosedLoop


def test_closed_loop_outputs():
    model = OneStepClosedLoop()

    x = torch.rand(2, 3, 64, 64)
    alpha = torch.tensor([0.25, 0.75])

    outputs = model(x, alpha)

    assert outputs["enhanced"].shape == x.shape
    assert outputs["redegraded"].shape == x.shape
    assert outputs["consequence_map"].shape == x.shape
    assert outputs["consequence_vector"].shape == (2, 128)
    assert outputs["updated_alpha"].shape == (2,)


def test_updated_alpha_range():
    model = OneStepClosedLoop()

    x = torch.rand(2, 3, 32, 32)
    alpha = torch.tensor([0.0, 1.0])

    outputs = model(x, alpha)

    assert torch.all(outputs["updated_alpha"] >= 0.0)
    assert torch.all(outputs["updated_alpha"] <= 1.0)


def test_alpha_zero_starts_from_original():
    model = OneStepClosedLoop()

    x = torch.rand(1, 3, 32, 32)
    alpha = torch.tensor([0.0])

    outputs = model(x, alpha)

    assert torch.allclose(
        outputs["enhanced"],
        x,
    )