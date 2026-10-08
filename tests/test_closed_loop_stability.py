import numpy as np
import pytest
import torch

from experiments.evaluate_closed_loop_stability import (
    CONTROLLER_PATH,
    DEVICE,
    ENHANCER_PATH,
    alpha_transition,
    detect_oscillation,
    feasible_action,
    l1_improvement,
    load_frozen_enhancer,
    load_controller_checkpoint,
    predictor,
    run_two_step_policy,
    total_improvement,
)


def test_step1_representation_is_recomputed_from_new_state():
    state_calls = []
    prediction_calls = []

    def state_at_alpha(alphas):
        alpha = float(alphas[0])
        current_image_stat = alpha * 10.0
        state_calls.append((alpha, current_image_stat))
        return np.asarray([[alpha, current_image_stat]], dtype=np.float32), {"q": np.asarray([1-alpha])}

    def predict(features, alphas):
        prediction_calls.append(features.copy())
        action = 1 if len(prediction_calls) == 1 else -1
        return [{"applied_action_int": action}]

    result = run_two_step_policy(
        np.asarray([0.5]), state_at_alpha, predict,
        lambda alphas: np.asarray([0.2]),
    )
    assert state_calls == [(0.5, 5.0), (0.6, 6.0)]
    assert result["representation1"][0, 1] == 6.0
    assert prediction_calls[1][0, 1] != prediction_calls[0][0, 1]


def test_step2_controller_prediction_is_recomputed_not_copied():
    calls = []

    def state_at_alpha(alphas):
        value = float(alphas[0])
        return np.asarray([[value]], dtype=np.float32), {"q": np.asarray([value])}

    def predict(features, alphas):
        calls.append((features.copy(), alphas.copy()))
        action = 1 if len(calls) == 1 else -1
        return [{"applied_action_int": action}]

    result = run_two_step_policy(np.asarray([0.4]), state_at_alpha, predict,
                                 lambda alphas: np.asarray([float(alphas[0])]))
    assert len(calls) == 2
    assert calls[0][0][0, 0] == 0.4
    assert calls[1][0][0, 0] == 0.5
    assert result["action0"].tolist() == [1]
    assert result["action1"].tolist() == [-1]
    assert result["alpha2"].tolist() == [0.4]


def test_alpha_transition_and_boundary_masking():
    assert alpha_transition(0.4, 1) == pytest.approx(0.5)
    assert alpha_transition(0.4, -1) == pytest.approx(0.3)
    assert alpha_transition(0.4, 0) == 0.4
    assert feasible_action("decrease", 0.0) == ("increase", False, True, True)
    assert feasible_action("increase", 1.0) == ("decrease", True, False, True)
    assert alpha_transition(0.05, -1) == 0.0


def test_incremental_and_total_improvement_signs():
    assert l1_improvement(0.4, 0.3) == pytest.approx(0.1)
    assert total_improvement(0.4, 0.2) == pytest.approx(0.2)
    assert l1_improvement(0.3, 0.4) == pytest.approx(-0.1)


def test_action_reversal_detection():
    assert detect_oscillation(1, -1)
    assert detect_oscillation(-1, 1)
    assert not detect_oscillation(1, 1)
    assert not detect_oscillation(0, -1)


def test_reconstructed_controller_checkpoint_and_threshold_load():
    saved, model = load_controller_checkpoint(CONTROLLER_PATH, device=DEVICE)
    assert saved["artifact_name"] == "reconstructed_baseline_b"
    assert model.in_features == 43 and model.out_features == 3
    assert saved["feature_mean"].shape == (1, 43)
    assert saved["feature_std"].shape == (1, 43)
    assert saved["threshold"] == 0.46343794465065
    assert saved["class_mapping"] == {"decrease": 0, "hold": 1, "increase": 2}


def test_controller_and_enhancer_are_frozen_and_inference_has_no_gradients():
    saved, controller = load_controller_checkpoint(CONTROLLER_PATH, device=DEVICE)
    before = {k: v.detach().clone() for k, v in controller.state_dict().items()}
    predict = predictor(controller, saved["feature_mean"].to(DEVICE),
                        saved["feature_std"].to(DEVICE), saved["threshold"],
                        saved["input_feature_order"])
    result = predict(np.zeros((2, 43), dtype=np.float32), np.asarray([0.5, 0.8]))
    assert len(result) == 2
    assert all(p.grad is None for p in controller.parameters())
    assert all(torch.equal(before[k], v) for k, v in controller.state_dict().items())

    _, enhancer = load_frozen_enhancer(ENHANCER_PATH, device=DEVICE)
    assert not enhancer.training
    assert all(not p.requires_grad and p.grad is None for p in enhancer.parameters())
    x = torch.zeros((1, 3, 8, 8), device=DEVICE)
    with torch.inference_mode():
        y = enhancer(x, torch.tensor([0.5], device=DEVICE))
    assert not y.requires_grad
