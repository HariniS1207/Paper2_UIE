import pytest
import torch

from src.models.control import validate_alpha, expand_alpha


def test_validate_alpha():
    alpha = torch.tensor([0.0, 0.5, 1.0])

    result = validate_alpha(alpha)

    assert torch.equal(result, alpha)


def test_validate_alpha_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_alpha(torch.tensor([-0.1]))

    with pytest.raises(ValueError):
        validate_alpha(torch.tensor([1.1]))


def test_expand_scalar_alpha():
    alpha = torch.tensor(0.5)

    result = expand_alpha(alpha, batch_size=4)

    assert result.shape == (4, 1)
    assert torch.allclose(result, torch.full((4, 1), 0.5))


def test_expand_batch_alpha():
    alpha = torch.tensor([0.0, 0.25, 0.5, 1.0])

    result = expand_alpha(alpha, batch_size=4)

    assert result.shape == (4, 1)
    assert torch.equal(result[:, 0], alpha)