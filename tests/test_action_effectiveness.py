import pytest
import torch

from experiments.evaluate_action_effectiveness import (
    MODELS, SOURCE, alpha_after, improvement, metrics, read_csv, threshold_action,
)


def test_alpha_update_and_hold():
    assert alpha_after(.4, 1) == pytest.approx(.5)
    assert alpha_after(.4, -1) == pytest.approx(.3)
    assert alpha_after(.4, 0) == pytest.approx(.4)


@pytest.mark.parametrize("alpha,action", [(0, -1), (1, 1)])
def test_infeasible_boundary_actions_raise_instead_of_clipping(alpha, action):
    with pytest.raises(ValueError, match="infeasible"):
        alpha_after(alpha, action)


def test_feasible_near_boundary_step_caps_at_endpoint():
    assert alpha_after(.05, -1) == 0
    assert alpha_after(.95, 1) == 1


def test_improvement_is_before_minus_after():
    assert improvement(.25, .2) == pytest.approx(.05)
    assert improvement(.2, .25) == pytest.approx(-.05)


def test_no_correction_is_exactly_zero():
    q = .123456
    assert improvement(q, q) == 0
    assert metrics([improvement(q, q)])['mean_improvement'] == 0


def test_oracle_and_controller_directions_apply_as_expected():
    # Existing oracle increase; saved controller decrease. Both map directly to +/-h.
    assert alpha_after(.5, {"increase": 1, "decrease": -1}["increase"]) == pytest.approx(.6)
    assert alpha_after(.5, {"increase": 1, "decrease": -1}["decrease"]) == pytest.approx(.4)


def test_saved_threshold_replay_matches_frozen_binary_output():
    thresholds = {r["model"]: float(r["threshold_on_conditional_probability_increase"])
                  for r in read_csv(SOURCE / "validation_thresholds.csv")}
    predictions = [r for r in read_csv(SOURCE / "predictions.csv")
                   if r["partition"] == "held-out-test" and r["model"] in set(MODELS.values())]
    for row in predictions:
        predicted = threshold_action(
            float(row["conditional_probability_increase"]), thresholds[row["model"]],
            row["valid_minus"] == "1", row["valid_plus"] == "1",
        )
        assert predicted == row["binary_prediction"]


def test_enhancer_can_be_frozen_without_gradients():
    model = torch.nn.Linear(2, 1)
    model.eval()
    model.requires_grad_(False)
    x = torch.ones(1, 2, requires_grad=True)
    with torch.inference_mode():
        y = model(x)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert not y.requires_grad
