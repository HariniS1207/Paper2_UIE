import numpy as np
import pytest
import torch

from experiments.diagnose_feedback_failure import image_stats
from experiments.run_local_probe_controller import (
    CLASS_ORDER,
    PHI_SUFFIXES,
    feature_groups,
    fit_classifier,
    generate_triplets,
    phi,
    probabilities,
    select_threshold,
    training_rows_only,
    triplet_alphas,
)


def test_phi_reuses_exact_14_dimensional_baseline_statistics():
    image = np.full((8, 8, 3), 0.25, dtype=np.float32)
    expected, _, _ = image_stats(image, "phi")
    assert len(PHI_SUFFIXES) == 14
    assert phi(image).shape == (14,)
    assert phi(image).tolist() == pytest.approx([expected[f"phi_{name}"] for name in PHI_SUFFIXES])


def test_representation_dimensions_and_c1_equals_baseline_b():
    groups = feature_groups()
    assert groups["baseline_B_alpha_current_statistics"] == groups["ablation_C1_current_state_only"]
    assert len(groups["baseline_A_alpha_only"]) == 1
    assert len(groups["baseline_B_alpha_current_statistics"]) == 43
    assert len(groups["candidate_C_full_local_response"]) == 73
    assert len(groups["ablation_C2_probe_response_only"]) == 31
    assert len(groups["ablation_C3_full_without_alpha"]) == 72


def test_boundary_probe_is_missing_and_not_fabricated():
    assert triplet_alphas(0.0, 0.0, 0.1, False, True) == {"current": 0.0, "plus": 0.1}
    assert triplet_alphas(1.0, 0.9, 1.0, True, False) == {"current": 1.0, "minus": 0.9}
    with pytest.raises(ValueError):
        triplet_alphas(0.0, 0.0, 0.1, True, True)


class AlphaOffset(torch.nn.Module):
    def forward(self, x, alpha):
        return x + alpha[:, None, None, None]


def test_triplets_use_expected_alpha_and_only_feasible_probe_at_boundary():
    x = torch.zeros((1, 3, 2, 2))
    boundary = [{"alpha": "0", "alpha_minus": "0", "alpha_plus": "0.1",
                 "valid_minus": "False", "valid_plus": "True"}]
    result = generate_triplets(AlphaOffset(), x, boundary, device=torch.device("cpu"))[0]
    assert set(result) == {"current", "plus"}
    assert np.allclose(result["current"], 0.0)
    assert np.allclose(result["plus"], 0.1)

    interior = [{"alpha": "0.5", "alpha_minus": "0.4", "alpha_plus": "0.6",
                 "valid_minus": "True", "valid_plus": "True"}]
    result = generate_triplets(AlphaOffset(), x, interior, device=torch.device("cpu"))[0]
    assert set(result) == {"minus", "current", "plus"}
    assert np.allclose(result["minus"], 0.4)
    assert np.allclose(result["current"], 0.5)
    assert np.allclose(result["plus"], 0.6)


def test_classifier_has_three_actions_and_train_only_class_weights():
    x = torch.tensor([[0.0], [0.2], [0.8], [1.0], [0.4], [0.6]])
    y = torch.tensor([0, 0, 0, 1, 2, 2])
    model, mean, scale, weights, counts = fit_classifier(x, y, epochs=2)
    assert model.out_features == len(CLASS_ORDER) == 3
    assert counts == [3, 1, 2]
    assert weights == pytest.approx([2 / 3, 2.0, 1.0])
    assert probabilities(model, mean, scale, x).shape == (6, 3)


def test_threshold_is_selected_from_validation_scores_with_feasibility():
    labels = ["increase", "increase", "decrease", "decrease"]
    scores = [0.45, 0.40, 0.30, 0.20]
    validation_rows = [{"partition": "controller-validation", "valid_minus": 1, "valid_plus": 1}
                       for _ in labels]
    assert select_threshold(labels, scores, validation_rows) == 0.4
    with pytest.raises(ValueError):
        select_threshold(labels, scores, [dict(row, partition="held-out-test") for row in validation_rows])


def test_training_rows_excludes_validation_and_heldout_pairs():
    rows = [
        {"partition": "controller-train", "image_id": "train"},
        {"partition": "controller-validation", "image_id": "validation"},
        {"partition": "held-out-test", "image_id": "test"},
    ]
    assert [row["image_id"] for row in training_rows_only(rows)] == ["train"]
