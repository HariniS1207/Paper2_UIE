import pytest

from src.training.feedback_oracle import choose_local_action


def test_oracle_chooses_only_candidate_that_improves_current_quality():
    result = choose_local_action(0.5, 0.2, 0.19, 0.21, h=0.1, margin=1e-4)
    assert result.direction == "decrease"
    assert result.delta == pytest.approx(-0.1)


def test_oracle_holds_when_both_candidates_are_worse():
    result = choose_local_action(0.5, 0.2, 0.21, 0.22, h=0.1, margin=1e-4)
    assert result.direction == "hold"
    assert result.delta == 0.0


def test_oracle_holds_within_margin():
    result = choose_local_action(0.5, 0.2, 0.19995, 0.19996, h=0.1, margin=1e-4)
    assert result.direction == "hold"


def test_oracle_holds_if_candidate_errors_are_near_tied():
    result = choose_local_action(0.5, 0.21, 0.20, 0.20005, h=0.1, margin=1e-4)
    assert result.direction == "hold"


def test_oracle_selects_candidate_that_beats_current_and_other_candidate():
    result = choose_local_action(0.5, 0.21, 0.205, 0.19, h=0.1, margin=1e-4)
    assert result.direction == "increase"


def test_alpha_zero_masks_clipped_duplicate_and_can_only_increase():
    result = choose_local_action(0.0, 0.2, 0.2, 0.1, h=0.1, margin=1e-4)
    assert not result.valid_minus
    assert result.valid_plus
    assert result.direction == "increase"
    assert result.delta == pytest.approx(0.1)


def test_alpha_one_masks_clipped_duplicate_and_can_only_decrease():
    result = choose_local_action(1.0, 0.2, 0.1, 0.2, h=0.1, margin=1e-4)
    assert result.valid_minus
    assert not result.valid_plus
    assert result.direction == "decrease"


def test_alpha_must_be_in_range():
    with pytest.raises(ValueError):
        choose_local_action(1.1, 0.2, 0.1, 0.1)
