import torch

from src.models.consequence_encoder import ConsequenceEncoder


def test_output_shapes():
    model = ConsequenceEncoder()

    original = torch.rand(2, 3, 64, 64)
    redegraded = torch.rand(2, 3, 64, 64)

    consequence, vector = model(original, redegraded)

    assert consequence.shape == (2, 3, 64, 64)
    assert vector.shape == (2, 128)


def test_consequence_is_nonnegative():
    model = ConsequenceEncoder()

    original = torch.rand(1, 3, 32, 32)
    redegraded = torch.rand(1, 3, 32, 32)

    consequence, _ = model(original, redegraded)

    assert torch.all(consequence >= 0)


def test_identical_images_have_zero_consequence():
    model = ConsequenceEncoder()

    original = torch.rand(1, 3, 32, 32)

    consequence, _ = model(original, original)

    assert torch.allclose(
        consequence,
        torch.zeros_like(consequence),
    )