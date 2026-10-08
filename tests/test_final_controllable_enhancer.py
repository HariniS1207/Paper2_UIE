import torch

from src.models.final_controllable_enhancer import FinalControllableEnhancer


def test_alpha_zero_is_exact_identity_and_outputs_are_bounded():
    model = FinalControllableEnhancer(features=8, blocks=1)
    x = torch.rand(2, 3, 32, 32)
    outputs = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        output = model(x, torch.full((2,), alpha))
        outputs.append(output)
        assert torch.all((output >= 0.0) & (output <= 1.0))
    assert torch.equal(outputs[0], x)
    assert not torch.allclose(outputs[0], outputs[-1])


def test_scalar_and_batched_alpha_shapes_and_validation():
    model = FinalControllableEnhancer(features=8, blocks=1)
    x = torch.rand(2, 3, 16, 16)
    assert model(x, torch.tensor(0.5)).shape == x.shape
    assert model(x, torch.tensor([0.25, 0.75])).shape == x.shape
    try:
        model(x, torch.tensor([0.2, 0.4, 0.6]))
    except ValueError:
        pass
    else:
        raise AssertionError("alpha batch mismatch should fail")


def test_checkpoint_state_load_and_deterministic_frozen_evaluation(tmp_path):
    torch.manual_seed(42)
    model = FinalControllableEnhancer(features=8, blocks=1).eval()
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    clone = FinalControllableEnhancer(features=8, blocks=1).eval()
    clone.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for parameter in clone.parameters():
        parameter.requires_grad_(False)
    x = torch.rand(1, 3, 16, 16)
    with torch.inference_mode():
        first = clone(x, torch.tensor([0.5]))
        second = clone(x, torch.tensor([0.5]))
    assert torch.equal(first, second)
    assert all(parameter.grad is None for parameter in clone.parameters())
    assert all(not parameter.requires_grad for parameter in clone.parameters())
