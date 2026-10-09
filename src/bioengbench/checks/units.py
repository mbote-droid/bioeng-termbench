"""Unit-carrying comparisons (Value Object pattern via pint).

Tolerances are declared as quantities, so the grader itself cannot compare
milliseconds with seconds by accident. Submission values arrive as plain
numbers whose units are fixed by the task contract (for example the key
`sdnn_ms`); `quantity` attaches that unit at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import pint

ureg = pint.UnitRegistry()
ureg.define("beat = [] = bt")  # dimensionless count, so beats/minute is a frequency
Quantity = ureg.Quantity


def quantity(value: float, unit: str) -> pint.Quantity:
    return Quantity(value, unit)


@dataclass(frozen=True)
class Tolerance:
    """Pass if |value - truth| <= max(absolute, relative * |truth|)."""

    absolute: pint.Quantity
    relative: float = 0.0

    def allowed(self, truth: pint.Quantity) -> pint.Quantity:
        rel: pint.Quantity = abs(truth) * self.relative
        absolute: pint.Quantity = self.absolute.to(truth.units)
        return max(absolute, rel)

    def within(self, value: pint.Quantity, truth: pint.Quantity) -> bool:
        error: pint.Quantity = abs(value.to(truth.units) - truth)
        return bool(error <= self.allowed(truth))
