from experiments.diagnose_feedback_failure import binary_metrics, bin_index


def test_alpha_one_is_in_last_decile():
    assert bin_index(1.0) == 9
    assert bin_index(0.1) == 1


def test_binary_metrics_report_decrease_recall_and_confusion():
    result = binary_metrics(
        ["increase", "increase", "decrease", "decrease"],
        [0.1, 0.8, 0.9, 0.2],
    )
    assert result["tn_increase"] == 1
    assert result["fp_increase_as_decrease"] == 1
    assert result["fn_decrease_as_increase"] == 1
    assert result["tp_decrease"] == 1
    assert result["decrease_recall"] == 0.5
    assert result["balanced_accuracy"] == 0.5
