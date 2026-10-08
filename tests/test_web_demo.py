import base64
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import (
    ALPHA_LEVELS,
    CONTROLLER_PATH,
    CONTROLLER_SHA256,
    ENHANCER_PATH,
    ENHANCER_SHA256,
    app,
    select_feasible_action,
    sha256_file,
    verify_checkpoint_hashes,
)


def png_bytes(size=(256, 256), color=(44, 103, 129)):
    image = Image.new("RGB", size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_checkpoint_hash_verification():
    assert verify_checkpoint_hashes() == {
        "enhancer": ENHANCER_SHA256,
        "controller": CONTROLLER_SHA256,
    }


def test_model_loading_is_frozen(client):
    service = app.state.inference
    assert service.enhancer.training is False
    assert service.controller.training is False
    assert all(not p.requires_grad for p in service.enhancer.parameters())
    assert all(not p.requires_grad for p in service.controller.parameters())
    assert sum(p.numel() for p in service.enhancer.parameters()) == 298_947
    assert service.controller.in_features == 43
    assert service.controller.out_features == 3


def test_alpha_zero_is_identity(client):
    data = png_bytes()
    response = client.post("/api/enhance", files={"image": ("input.png", data, "image/png")}, data={"alpha": "0"})
    assert response.status_code == 200
    result = Image.open(io.BytesIO(base64.b64decode(response.json()["enhanced_image"]))).convert("RGB")
    with Image.open(io.BytesIO(data)) as source:
        assert np.array_equal(np.asarray(result), np.asarray(source.convert("RGB")))


def test_enhancer_output_is_in_range(client):
    service = app.state.inference
    tensor = torch.rand(1, 3, 256, 256, device=service.device)
    with torch.inference_mode():
        output = service.enhancer(tensor, torch.tensor([0.5], device=service.device))
    assert torch.isfinite(output).all()
    assert output.min().item() >= 0
    assert output.max().item() <= 1


def test_manual_alpha_levels_are_exact_presets():
    assert ALPHA_LEVELS == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_controller_has_three_logits(client):
    service = app.state.inference
    with torch.inference_mode():
        logits = service.controller(torch.zeros(1, 43, device=service.device))
    assert logits.shape == (1, 3)


def test_feasible_action_masking_at_boundaries():
    assert select_feasible_action([0.99, 0.005, 0.005], 0.0) == "hold"
    assert select_feasible_action([0.005, 0.99, 0.005], 1.0) == "hold"
    assert select_feasible_action([0.99, 0.005, 0.005], 0.01) == "decrease"
    assert select_feasible_action([0.005, 0.005, 0.99], 0.99) == "increase"


@pytest.mark.parametrize("alpha,expected", [(0.0, {"hold", "increase"}), (1.0, {"decrease", "hold"})])
def test_controller_alpha_boundary_behavior(client, alpha, expected):
    response = client.post("/api/controller", files={"image": ("input.png", png_bytes(), "image/png")}, data={"alpha": str(alpha)})
    assert response.status_code == 200
    assert response.json()["action"] in expected
    assert response.json()["suggested_alpha"] == pytest.approx(
        max(0.0, min(1.0, alpha + {"decrease": -0.1, "hold": 0.0, "increase": 0.1}[response.json()["action"]]))
    )


def test_invalid_image_is_rejected(client):
    response = client.post("/api/enhance", files={"image": ("broken.png", b"not an image", "image/png")}, data={"alpha": "0.5"})
    assert response.status_code == 400


def test_manual_api_response_schema(client):
    response = client.post("/api/enhance", files={"image": ("input.png", png_bytes((320, 180)), "image/png")}, data={"alpha": "0.25"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"original_image", "enhanced_image", "alpha", "metadata"}
    assert body["alpha"] == 0.25
    assert body["metadata"]["reference_available"] is False
    assert (body["metadata"]["input_width"], body["metadata"]["input_height"]) == (320, 180)


def test_alpha_comparison_api_returns_every_preset(client):
    response = client.post("/api/compare-alphas", files={"image": ("input.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert set(body["outputs"]) == {f"{alpha:.2f}" for alpha in ALPHA_LEVELS}
    assert body["alphas"] == list(ALPHA_LEVELS)
    assert body["metadata"]["reference_available"] is False


def test_controller_probabilities_sum_to_one(client):
    response = client.post("/api/controller", files={"image": ("input.png", png_bytes(), "image/png")}, data={"alpha": "0.5"})
    assert response.status_code == 200
    body = response.json()
    assert set(body["probabilities"]) == {"decrease", "hold", "increase"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["recommendation_label"] == "Controller recommendation"


def test_inference_is_deterministic(client):
    data = png_bytes()
    kwargs = {"files": {"image": ("input.png", data, "image/png")}, "data": {"alpha": "0.75"}}
    first = client.post("/api/enhance", **kwargs).json()["enhanced_image"]
    second = client.post("/api/enhance", **kwargs).json()["enhanced_image"]
    assert first == second


def test_checkpoint_files_remain_unchanged_after_inference(client):
    before = {"enhancer": sha256_file(ENHANCER_PATH), "controller": sha256_file(CONTROLLER_PATH)}
    response = client.post("/api/enhance", files={"image": ("input.png", png_bytes(), "image/png")}, data={"alpha": "0.5"})
    assert response.status_code == 200
    after = {"enhancer": sha256_file(ENHANCER_PATH), "controller": sha256_file(CONTROLLER_PATH)}
    assert before == after == {"enhancer": ENHANCER_SHA256, "controller": CONTROLLER_SHA256}
