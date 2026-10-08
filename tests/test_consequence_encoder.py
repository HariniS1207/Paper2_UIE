import torch

from src.models.consequence_encoder import ConsequenceEncoder


def test_output_shapes():
    model = ConsequenceEncoder()

    original = torch.rand(2, 3, 64, 64)
    enhanced = torch.rand(2, 3, 64, 64)
    alpha = torch.tensor([0.2, 0.8])

    consequence, vector = model(original, enhanced, alpha)

    assert consequence.shape == (2, 10, 64, 64)
    assert vector.shape == (2, 128)


def test_state_contains_signed_enhancement_residual():
    model = ConsequenceEncoder()

    original = torch.zeros(1, 3, 32, 32)
    enhanced = torch.ones(1, 3, 32, 32) * 0.4

    consequence, _ = model(original, enhanced, torch.tensor([0.5]))

    assert torch.allclose(consequence[:, :3], original)
    assert torch.allclose(consequence[:, 3:6], enhanced)
    assert torch.allclose(consequence[:, 6:9], enhanced - original)
    assert torch.allclose(consequence[:, 9], torch.full((1, 32, 32), 0.5))


def test_equal_input_and_output_have_zero_residual():
    model = ConsequenceEncoder()

    original = torch.rand(1, 3, 32, 32)

    consequence, _ = model(original, original, torch.tensor([0.5]))

    assert torch.allclose(
        consequence[:, 6:9],
        torch.zeros_like(consequence[:, 6:9]),
    )
