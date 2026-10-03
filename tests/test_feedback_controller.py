import torch

from src.models.feedback_controller import FeedbackController


def test_output_shape():
    model = FeedbackController()

    consequence = torch.rand(4, 128)
    alpha = torch.tensor([0.0, 0.25, 0.5, 1.0])

    updated_alpha = model(consequence, alpha)

    assert updated_alpha.shape == (4,)


def test_output_range():
    model = FeedbackController()

    consequence = torch.rand(4, 128)
    alpha = torch.tensor([0.0, 0.25, 0.5, 1.0])

    updated_alpha = model(consequence, alpha)

    assert torch.all(updated_alpha >= 0.0)
    assert torch.all(updated_alpha <= 1.0)


def test_scalar_alpha():
    model = FeedbackController()

    consequence = torch.rand(3, 128)
    alpha = torch.tensor(0.5)

    updated_alpha = model(consequence, alpha)

    assert updated_alpha.shape == (3,)