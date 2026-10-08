import hashlib

import numpy as np
import pytest
import torch

from experiments.train_final_baseline_b_phase3 import (
    ACTION_DIRECTION,
    ACTION_INDEX,
    CLASS_ORDER,
    EXPECTED_ENHANCER_SHA,
    ENHANCER_PATH,
    apply_action,
    fit_linear,
    load_frozen_enhancer,
    predict,
    state_partitions,
)
from src.training.feedback_oracle import choose_local_action


def test_baseline_b_dimensions_and_action_order_are_fixed():
    assert CLASS_ORDER == ("decrease", "hold", "increase")
    assert ACTION_INDEX == {"decrease": 0, "hold": 1, "increase": 2}
    assert ACTION_DIRECTION == {"decrease": -1, "hold": 0, "increase": 1}


def test_oracle_respects_feasibility_margin_and_near_ties():
    assert choose_local_action(0.0, 0.4, 0.0, 0.39).direction == "increase"
    assert choose_local_action(1.0, 0.4, 0.39, 0.0).direction == "decrease"
    assert choose_local_action(0.0, 0.4, -100.0, 0.39995).direction == "hold"
    assert choose_local_action(0.5, 0.4, 0.39, 0.39005).direction == "hold"


def test_alpha_action_update_respects_boundaries_and_rejects_unknown_action():
    assert apply_action(0.05, "decrease") == 0.0
    assert apply_action(0.95, "increase") == 1.0
    assert apply_action(0.5, "hold") == 0.5
    with pytest.raises(ValueError):
        apply_action(0.5, "sideways")


def test_state_partitioning_never_includes_test_in_development():
    states = [
        {"partition": "controller-train", "id": "tr"},
        {"partition": "controller-validation", "id": "va"},
        {"partition": "held-out-test", "id": "te"},
    ]
    parts = state_partitions(states)
    development = parts["controller-train"] + parts["controller-validation"]
    assert [r["id"] for r in development] == ["tr", "va"]
    assert all(r["partition"] != "held-out-test" for r in development)


def test_training_only_normalization_weights_and_test_rows_rejected():
    rows = []
    for label, offset in zip(CLASS_ORDER, (0.0, 10.0, 20.0)):
        for j in range(2):
            feat = np.full(43, offset + j, dtype=np.float32)
            rows.append({"partition": "controller-train", "oracle_action": label, "features": feat})
    model, mean, std, scale, counts, weights, _ = fit_linear(rows, 43)
    expected = np.stack([r["features"] for r in rows]).mean(axis=0)
    assert model.in_features == 43 and model.out_features == 3
    assert torch.allclose(mean, torch.tensor(expected))
    assert torch.equal(counts, torch.tensor([2, 2, 2]))
    assert torch.allclose(weights, torch.ones(3))
    assert torch.all(scale >= 1e-6)
    rows[0]["partition"] = "held-out-test"
    with pytest.raises(ValueError):
        fit_linear(rows, 43)


def test_controller_checkpoint_load_and_deterministic_inference(tmp_path):
    rows = []
    for label, offset in zip(CLASS_ORDER, (0.0, 10.0, 20.0)):
        for j in range(2):
            rows.append({"partition": "controller-train", "oracle_action": label,
                         "features": np.full(43, offset + j, dtype=np.float32)})
    model, mean, std, scale, *_ = fit_linear(rows, 43)
    path = tmp_path / "controller.pth"
    torch.save({"model_state_dict": model.state_dict(), "mean": mean, "scale": scale}, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    clone = torch.nn.Linear(43, 3).eval()
    clone.load_state_dict(saved["model_state_dict"], strict=True)
    x = torch.from_numpy(np.stack([r["features"] for r in rows]))
    with torch.inference_mode():
        p1 = torch.softmax(model((x - mean) / scale), 1)
        p2 = torch.softmax(clone((x - saved["mean"]) / saved["scale"]), 1)
    assert torch.equal(p1, p2)
    assert torch.isfinite(p1).all()


def test_phase2_enhancer_checkpoint_is_strictly_loaded_and_frozen():
    assert hashlib.sha256(ENHANCER_PATH.read_bytes()).hexdigest().upper() == EXPECTED_ENHANCER_SHA
    model, digest = load_frozen_enhancer()
    assert digest == EXPECTED_ENHANCER_SHA
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    device = next(model.parameters()).device
    x = torch.rand(1, 3, 16, 16, device=device)
    with torch.inference_mode():
        assert torch.equal(model(x, torch.tensor([0.0], device=device)), x)
