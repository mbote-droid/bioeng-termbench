"""Event matching (for example detected vs true R peaks) in O(n + m).

The old matcher compared every true event with every detection, O(n*m).
Here both lists are sorted and walked once with two pointers.

Greedy two-pointer matching is optimal (maximum number of matched pairs) when
events within each list are separated by more than twice the tolerance,
because then no detection can be within tolerance of two true events. For
R peaks, physiology guarantees spacing >= 250 ms and the tolerance is 40 ms.
Callers outside that regime must not use this function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class MatchCounts:
    true_positive: int
    false_negative: int
    false_positive: int

    @property
    def sensitivity(self) -> float:
        n = self.true_positive + self.false_negative
        return self.true_positive / n if n else 1.0

    @property
    def precision(self) -> float:
        n = self.true_positive + self.false_positive
        return self.true_positive / n if n else 1.0


def match_events(truth: ArrayLike, detected: ArrayLike, tol: int) -> MatchCounts:
    t = np.sort(np.asarray(truth, dtype=np.int64))
    d = np.sort(np.asarray(detected, dtype=np.int64))
    i = j = tp = 0
    while i < len(t) and j < len(d):
        gap = d[j] - t[i]
        if abs(gap) <= tol:
            tp += 1
            i += 1
            j += 1
        elif gap < 0:
            j += 1  # detection before this truth event and too far: false positive
        else:
            i += 1  # truth event with no detection near it: false negative
    return MatchCounts(tp, len(t) - tp, len(d) - tp)
