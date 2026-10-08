import torch

from src.models.feedback_controller import FeedbackController
from src.models.feedback_controller import AlphaOnlyController, ImageStateController


def test_adjustment_shape_and_bound():
    model = FeedbackController()

    consequence = torch.rand(4, 128)
    alpha = torch.tensor([0.0, 0.25, 0.5, 1.0])

    adjustment = model(consequence, alpha)

    assert adjustment.shape == (4,)
    assert torch.all(adjustment >= -0.10)
    assert torch.all(adjustment <= 0.10)


def test_scalar_alpha():
    model = FeedbackController()

    consequence = torch.rand(3, 128)
    alpha = torch.tensor(0.5)

    adjustment = model(consequence, alpha)

    assert adjustment.shape == (3,)


def test_alpha_only_baseline_is_bounded():
    model = AlphaOnlyController()
    result = model(torch.tensor([0.0, 0.5, 1.0]))
    assert result.shape == (3,)
    assert torch.all(result.abs() <= 0.10)


def test_image_state_baseline_is_bounded():
    model = ImageStateController()
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand_like(x)
    result = model(x, y, torch.tensor([0.2, 0.8]))
    assert result.shape == (2,)
    assert torch.all(result.abs() <= 0.10)
