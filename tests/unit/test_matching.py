import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import linear_sum_assignment

from bioengbench.checks.matching import match_events

TOL = 14


def _optimal_tp(truth: np.ndarray, det: np.ndarray, tol: int) -> int:
    """Maximum matching via the Hungarian algorithm (reference, O(n^3))."""
    if len(truth) == 0 or len(det) == 0:
        return 0
    cost = (np.abs(truth[:, None] - det[None, :]) > tol).astype(float)
    rows, cols = linear_sum_assignment(cost)
    return int(np.sum(cost[rows, cols] == 0))


def _spaced(draw_list: list[int], min_gap: int) -> np.ndarray:
    return np.cumsum(np.asarray(draw_list) + min_gap)


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.integers(0, 300), max_size=40),
    st.lists(st.integers(0, 300), max_size=40),
)
def test_greedy_equals_optimal_when_events_are_spaced(gaps_t: list[int], gaps_d: list[int]) -> None:
    # Precondition from the module docstring: spacing within each list > 2 * tol.
    truth = _spaced(gaps_t, 2 * TOL + 1)
    det = _spaced(gaps_d, 2 * TOL + 1)
    counts = match_events(truth, det, TOL)
    assert counts.true_positive == _optimal_tp(truth, det, TOL)
    assert counts.true_positive + counts.false_negative == len(truth)
    assert counts.true_positive + counts.false_positive == len(det)


def test_perfect_and_shifted() -> None:
    t = np.arange(10) * 300
    assert match_events(t, t, TOL).sensitivity == 1.0
    assert match_events(t, t + TOL + 1, TOL).true_positive == 0
    assert match_events(t, t + TOL, TOL).true_positive == 10


def test_runs_in_linear_time_on_large_inputs() -> None:
    t = np.arange(200_000) * 300
    counts = match_events(t, t + 3, TOL)  # an O(n*m) matcher would not finish quickly
    assert counts.true_positive == 200_000
