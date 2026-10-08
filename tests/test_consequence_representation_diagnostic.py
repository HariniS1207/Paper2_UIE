import numpy as np

from experiments.diagnose_consequence_representation import score, select_threshold


def test_validation_threshold_uses_balanced_accuracy_not_fixed_half():
    labels = ["increase", "increase", "decrease", "decrease"]
    probabilities = np.asarray([0.45, 0.4, 0.3, 0.2])
    threshold = select_threshold(labels, probabilities)

    assert threshold == 0.4
    assert score(labels, probabilities, threshold)["balanced_accuracy"] == 1.0


def test_score_reports_decrease_confusion_and_recall():
    result = score(
        ["increase", "increase", "decrease", "decrease"],
        np.asarray([0.9, 0.2, 0.8, 0.1]),
        threshold=0.5,
    )

    assert result["tn_decrease_as_decrease"] == 1
    assert result["fp_decrease_as_increase"] == 1
    assert result["fn_increase_as_decrease"] == 1
    assert result["tp_increase"] == 1
    assert result["decrease_recall"] == 0.5
