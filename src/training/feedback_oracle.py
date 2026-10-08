"""Paired-quality local action oracle with feasible-boundary handling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleDecision:
    direction: str
    delta: float
    gain_minus: float
    gain_plus: float
    valid_minus: bool
    valid_plus: bool


def choose_local_action(alpha, q_current, q_minus, q_plus, h=0.10, margin=1e-4):
    """Choose the best feasible action only when it improves current L1 by margin."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if h <= 0 or margin < 0:
        raise ValueError("h must be positive and margin nonnegative")
    a_minus, a_plus = max(0.0, alpha - h), min(1.0, alpha + h)
    valid_minus, valid_plus = a_minus < alpha, a_plus > alpha
    gain_minus = q_current - q_minus if valid_minus else float("-inf")
    gain_plus = q_current - q_plus if valid_plus else float("-inf")
    candidates = []
    if valid_minus:
        candidates.append((gain_minus, "decrease", a_minus - alpha))
    if valid_plus:
        candidates.append((gain_plus, "increase", a_plus - alpha))
    if not candidates:
        return OracleDecision("hold", 0.0, gain_minus, gain_plus, valid_minus, valid_plus)
    gain, direction, delta = max(candidates, key=lambda candidate: candidate[0])
    if gain <= margin:
        return OracleDecision("hold", 0.0, gain_minus, gain_plus, valid_minus, valid_plus)
    if len(candidates) == 2:
        other_gain = min(gain_minus, gain_plus)
        if gain - other_gain <= margin:
            return OracleDecision("hold", 0.0, gain_minus, gain_plus, valid_minus, valid_plus)
    return OracleDecision(direction, delta, gain_minus, gain_plus, valid_minus, valid_plus)
