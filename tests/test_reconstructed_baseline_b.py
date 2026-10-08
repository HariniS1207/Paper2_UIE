import csv
import json

import numpy as np
import torch

from experiments.reconstruct_baseline_b import (
    CLASS_ORDER,
    FEATURES,
    MODEL_NAME,
    OUT,
    ORIGINAL_PREDICTIONS,
    ORIGINAL_THRESHOLDS,
    load_controller_checkpoint,
)
from experiments.run_local_probe_controller import (
    binary_predictions,
    conditional_increase_probability,
    feature_groups,
    probabilities,
    read_csv,
)


def test_reconstructed_checkpoint_loads_with_required_inference_state():
    saved, model = load_controller_checkpoint(OUT / "controller.pth")
    assert model.in_features == 43
    assert model.out_features == 3
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert saved["artifact_name"] == "reconstructed_baseline_b"
    assert saved["original_checkpoint_recoverable"] is False


def test_normalization_and_feature_order_are_persisted():
    saved, _ = load_controller_checkpoint(OUT / "controller.pth")
    assert tuple(saved["feature_mean"].shape) == (1, 43)
    assert tuple(saved["feature_std"].shape) == (1, 43)
    assert torch.isfinite(saved["feature_mean"]).all()
    assert torch.isfinite(saved["feature_std"]).all()
    assert torch.all(saved["feature_std"] >= 1e-6)
    assert saved["input_feature_order"] == feature_groups()[MODEL_NAME]


def test_class_mapping_and_threshold_are_persisted():
    saved, _ = load_controller_checkpoint(OUT / "controller.pth")
    assert saved["class_order"] == list(CLASS_ORDER)
    assert saved["class_mapping"] == {"decrease": 0, "hold": 1, "increase": 2}
    original = [r for r in read_csv(ORIGINAL_THRESHOLDS) if r["model"] == MODEL_NAME]
    assert len(original) == 1
    assert saved["threshold"] == float(original[0]["threshold_on_conditional_probability_increase"])


def test_loaded_reconstruction_replays_saved_test_predictions_deterministically():
    saved, model = load_controller_checkpoint(OUT / "controller.pth")
    columns = saved["input_feature_order"]
    rows = [r for r in read_csv(FEATURES) if r["partition"] == "held-out-test"][:16]
    x = torch.tensor([[float(r[c]) for c in columns] for r in rows], dtype=torch.float32)
    first = probabilities(model, saved["feature_mean"], saved["feature_std"], x)
    second = probabilities(model, saved["feature_mean"], saved["feature_std"], x)
    np.testing.assert_array_equal(first, second)
    p_inc = conditional_increase_probability(first, rows)
    actions = binary_predictions(p_inc, saved["threshold"], rows)
    with (OUT / "test_predictions.csv").open(newline="", encoding="utf-8") as f:
        persisted = list(csv.DictReader(f))[:16]
    assert len(persisted) == len(rows)
    assert [str(a) for a in actions] == [r["binary_prediction"] for r in persisted]


def test_reconstructed_predictions_are_compared_to_original_baseline_b():
    original = [r for r in read_csv(ORIGINAL_PREDICTIONS)
                if r["model"] == MODEL_NAME and r["partition"] == "held-out-test"]
    reconstruction = read_csv(OUT / "test_predictions.csv")
    assert len(original) == len(reconstruction) == 1748
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["reconstruction_status"] == "VERIFIED"
    assert metrics["test"]["class_prediction_agreement_rate"] == 1.0
    assert metrics["test"]["thresholded_binary_agreement_rate"] == 1.0
