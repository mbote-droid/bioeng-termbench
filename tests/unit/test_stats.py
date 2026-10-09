import math

import pytest

from bioengbench.checks.stats import wilson_interval, zero_failure_upper_bound


def test_wilson_known_value() -> None:
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


@pytest.mark.parametrize(("k", "n"), [(0, 5), (5, 5), (0, 1), (3, 50)])
def test_wilson_stays_in_unit_interval(k: int, n: int) -> None:
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


@pytest.mark.parametrize("n", [10, 30, 100, 1000])
def test_rule_of_three(n: int) -> None:
    exact = zero_failure_upper_bound(n)
    assert exact == pytest.approx(3 / n, rel=0.15)
    assert (1 - exact) ** n == pytest.approx(0.05)


def test_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        zero_failure_upper_bound(0)
    assert math.isfinite(zero_failure_upper_bound(1))
