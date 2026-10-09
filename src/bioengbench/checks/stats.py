"""Confidence bounds for grader error rates.

Grader validity is reported as binomial proportions: the false-accept rate
(wrong solutions that pass) and the false-reject rate (correct solutions
that fail). Point estimates from small n are misleading, so every reported
rate carries an interval.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054  # two-sided 95 % standard-normal quantile


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    p_hat = k/n,  centre = (p_hat + z^2/2n) / (1 + z^2/n),
    half  = z * sqrt(p_hat(1-p_hat)/n + z^2/4n^2) / (1 + z^2/n).

    Unlike the Wald interval it stays inside [0, 1] and behaves at k = 0 or n.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need n > 0 and 0 <= successes <= n")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def zero_failure_upper_bound(n: int, confidence: float = 0.95) -> float:
    """Exact upper bound on a rate after observing 0 events in n trials.

    Solves (1 - p)^n = 1 - confidence, so p = 1 - (1 - confidence)^(1/n).
    For 95 % this is about 3/n (the "rule of three").
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return float(1.0 - (1.0 - confidence) ** (1.0 / n))
