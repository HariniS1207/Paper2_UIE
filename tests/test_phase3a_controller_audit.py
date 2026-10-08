import numpy as np
import pytest

from experiments.audit_phase3a_controller import (
    paired_bootstrap,
    distribution,
    classification_metrics,
    benefit_stats,
    wilcoxon_signed_rank,
)


def test_paired_bootstrap_is_deterministic_and_uses_difference_vector():
    differences = np.array([0.2, -0.1, 0.0, 0.4, -0.05])
    first = paired_bootstrap(differences, samples=500, seed=42)
    second = paired_bootstrap(differences, samples=500, seed=42)
    assert first == second
    assert first["mean"] == pytest.approx(float(differences.mean()))
    assert first["paired_unit"] == "same held-out controller state"
    assert first["ci95_percentile"][0] <= first["mean"] <= first["ci95_percentile"][1]


def test_distribution_reports_median_iqr_and_requested_quantiles():
    values = np.arange(1, 11, dtype=float)
    stats = distribution(values)
    assert stats["median"] == 5.5
    assert stats["p25"] == pytest.approx(3.25)
    assert stats["p75"] == pytest.approx(7.75)
    assert stats["iqr"] == pytest.approx(4.5)
    assert stats["p10"] < stats["p25"] < stats["p50"] < stats["p75"] < stats["p90"]


def test_action_metrics_preserve_decrease_hold_increase_order():
    confusion, metrics = classification_metrics(
        ["decrease", "decrease", "hold", "increase", "increase"],
        ["decrease", "hold", "hold", "decrease", "increase"],
    )
    assert confusion == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["per_class"]["decrease"]["precision"] == pytest.approx(0.5)
    assert metrics["per_class"]["decrease"]["recall"] == pytest.approx(0.5)


def test_benefit_rates_partition_observations():
    result = benefit_stats([0.1, 0.0, -0.1, 1e-9])
    assert result["improved_pct"] == 25.0
    assert result["worsened_pct"] == 25.0
    assert result["unchanged_pct"] == 50.0


def test_bootstrap_rejects_empty_and_nonfinite_values():
    with pytest.raises(ValueError):
        paired_bootstrap([])
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, float("nan")])


def test_signed_rank_documents_zero_and_tied_rank_handling():
    result = wilcoxon_signed_rank([1.0, 1.0, -1.0, 0.0])
    assert result["zero_count"] == 1
    assert result["nonzero_n"] == 3
    assert result["w_plus"] == 4.0
    assert result["w_minus"] == 2.0
    assert 0.0 <= result["p_value"] <= 1.0
    all_tied = wilcoxon_signed_rank([0.0, 0.0])
    assert all_tied["p_value"] == 1.0
