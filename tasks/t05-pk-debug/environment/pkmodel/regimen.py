"""Dosing regimen input."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Dose:
    start_h: float
    duration_h: float
    amount_mg: float

    @property
    def is_bolus(self) -> bool:
        return self.duration_h == 0.0


def load_regimen(path):
    """Return (doses sorted by start time, output grid dict)."""
    with open(path) as fh:
        raw = json.load(fh)
    by_time = {d["start_h"]: Dose(d["start_h"], d["duration_h"], d["amount_mg"]) for d in raw["doses"]}
    for d in by_time.values():
        if d.duration_h < 0 or d.amount_mg < 0 or d.start_h < 0:
            raise ValueError(f"invalid dose {d}")
    return [by_time[t] for t in sorted(by_time)], raw["output"]
