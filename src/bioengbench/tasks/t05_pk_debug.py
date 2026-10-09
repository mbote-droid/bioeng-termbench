"""t05: debug a broken pharmacokinetics package.

The agent receives `pkmodel`, a small package used by a (fictional) ward
dosing calculator, with a written specification (README.md), a partial test
suite and two user bug reports. It contains five defects, each a classic
failure in scientific software:

  1. units       CL arrives in mL/min and is converted the wrong way
  2. integrator  an adaptive explicit solver over the whole horizon with a
                 loose tolerance: inaccurate, and it can step over a short
                 infusion entirely (the rate is discontinuous in time)
  3. bolus       IV boluses (duration 0) are silently skipped
  4. precision   the output grid is built in float32, so long grids drift
  5. data model  doses are keyed by start time, so two doses given together
                 collapse into one

Only two are visible from the bug reports and the existing tests; the others
must be found by reading the specification against the code. The grader runs
`python -m pkmodel.cli` (the submission is the package directory, put on
PYTHONPATH) on five hidden scenarios per case, each aimed at one defect,
and compares every output with the exact solution (matrix exponential with
bolus jumps). A defect can be fixed any way that meets the specification.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from bioengbench.checks.base import CheckContext, CheckGroup, FunctionCheck, Outcome
from bioengbench.checks.io import FloatArray, read_csv_columns
from bioengbench.core.registry import register
from bioengbench.core.seeds import streams
from bioengbench.core.task import RunPlan, TaskPlugin
from bioengbench.models.pk import rate_matrix

GRID_TOL_H = 1e-9
CONC_TOL_REL = 1e-5  # of the peak; the specification promises 1e-6
COLUMNS = ("time_h", "conc_mg_per_L")


@dataclass(frozen=True)
class Dose:
    start_h: float
    duration_h: float  # 0 = bolus
    amount_mg: float


@dataclass(frozen=True)
class Scenario:
    name: str
    code: str
    params: dict[str, float]  # as the package reads them (CL in mL/min)
    doses: tuple[Dose, ...]
    grid: tuple[float, float, float]  # start, stop, step

    def times(self) -> FloatArray:
        start, stop, step = self.grid
        n = round((stop - start) / step)
        out: FloatArray = start + step * np.arange(n + 1, dtype=np.float64)
        return out


@dataclass(frozen=True)
class DebugCase:
    seed: int
    scenarios: tuple[Scenario, ...]


def exact_concentrations(
    params: dict[str, float], doses: tuple[Dose, ...], t: FloatArray
) -> FloatArray:
    """Exact solution: piecewise matrix exponential, boluses as jumps in A1,
    right-continuous at bolus times."""
    from scipy.linalg import expm  # noqa: PLC0415

    p = np.array(
        [
            params["CL_mL_per_min"] * 60.0 / 1000.0,
            params["V1_L"],
            params["Q_L_per_h"],
            params["V2_L"],
        ]
    )
    k = rate_matrix(p)
    infusions = [d for d in doses if d.duration_h > 0]
    events = {0.0, *t.tolist(), *(d.start_h for d in doses)}
    events |= {d.start_h + d.duration_h for d in infusions}
    grid = sorted(events)

    def rate(at: float) -> float:
        return sum(
            d.amount_mg / d.duration_h
            for d in infusions
            if d.start_h <= at < d.start_h + d.duration_h
        )

    def bolus(at: float) -> float:
        return sum(d.amount_mg for d in doses if d.duration_h == 0 and d.start_h == at)

    x = np.array([bolus(0.0), 0.0])
    values = {0.0: x[0] / p[1]}
    for a, b in itertools.pairwise(grid):
        m = np.zeros((3, 3))
        m[:2, :2] = k
        m[0, 2] = rate(0.5 * (a + b))
        e = expm(m * (b - a))
        x = e[:2, :2] @ x + e[:2, 2]
        x[0] += bolus(b)
        values[b] = x[0] / p[1]
    return np.array([values[ti] for ti in t.tolist()])


def _params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "CL_mL_per_min": float(rng.uniform(50.0, 120.0)),
        "V1_L": float(rng.uniform(10.0, 25.0)),
        "Q_L_per_h": float(rng.uniform(5.0, 20.0)),
        "V2_L": float(rng.uniform(20.0, 60.0)),
    }


@lru_cache(maxsize=64)
def make_case(seed: int) -> DebugCase:
    r = streams(seed, 5)
    amount = float(r[0].uniform(500.0, 1000.0))
    standard = Scenario(
        "standard",
        "STANDARD",
        _params(r[0]),
        tuple(Dose(12.0 * i, 1.0, amount) for i in range(3)),
        (0.0, 48.0, 0.25),
    )
    late = float(r[1].uniform(28.0, 34.0))
    short = Scenario(
        "short_infusion",
        "SHORT_INFUSION",
        _params(r[1]),
        (
            Dose(0.0, 1.0, float(r[1].uniform(500.0, 1000.0))),
            Dose(late, float(r[1].uniform(0.1, 0.3)), float(r[1].uniform(300.0, 600.0))),
        ),
        (0.0, 48.0, 2.0),
    )
    bolus = Scenario(
        "bolus",
        "BOLUS",
        _params(r[2]),
        (
            Dose(0.0, 0.0, float(r[2].uniform(300.0, 600.0))),
            Dose(float(r[2].uniform(4.0, 8.0)), 1.0, float(r[2].uniform(300.0, 600.0))),
        ),
        (0.0, 24.0, 0.5),
    )
    daily = float(r[3].uniform(500.0, 1000.0))
    long = Scenario(
        "long_horizon",
        "LONG_HORIZON",
        _params(r[3]),
        tuple(Dose(24.0 * i, 1.0, daily) for i in range(14)),
        (0.0, 336.0, 0.1),
    )
    coincident = Scenario(
        "coincident",
        "COINCIDENT_DOSES",
        _params(r[4]),
        (
            Dose(0.0, 0.5, float(r[4].uniform(800.0, 1200.0))),  # loading
            Dose(0.0, 4.0, float(r[4].uniform(150.0, 300.0))),  # maintenance, same start
            Dose(8.0, 4.0, float(r[4].uniform(150.0, 300.0))),
        ),
        (0.0, 24.0, 0.25),
    )
    return DebugCase(seed, (standard, short, bolus, long, coincident))


def write_inputs(case: DebugCase, input_dir: Path) -> None:
    for s in case.scenarios:
        d = input_dir / s.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "params.json").write_text(json.dumps(s.params, indent=2) + "\n")
        regimen = {
            "doses": [
                {"start_h": x.start_h, "duration_h": x.duration_h, "amount_mg": x.amount_mg}
                for x in s.doses
            ],
            "output": {"start_h": s.grid[0], "stop_h": s.grid[1], "step_h": s.grid[2]},
        }
        (d / "regimen.json").write_text(json.dumps(regimen, indent=2) + "\n")


@lru_cache(maxsize=512)
def _reference(seed: int, name: str) -> FloatArray:
    s = next(x for x in make_case(seed).scenarios if x.name == name)
    return exact_concentrations(s.params, s.doses, s.times())


def scenario_check(name: str, code: str) -> FunctionCheck:
    def fn(ctx: CheckContext) -> Outcome:
        case: DebugCase = ctx.truth
        scn = next(s for s in case.scenarios if s.name == name)
        out = read_csv_columns(ctx.run_dirs[name] / "conc.csv", COLUMNS)
        t_ref = scn.times()
        t = out["time_h"]
        if t.size != t_ref.size:
            return Outcome(False, f"{t.size} output times, expected {t_ref.size}")
        grid_err = float(np.max(np.abs(t - t_ref)))
        if grid_err > GRID_TOL_H:
            return Outcome(
                False, f"output times off by up to {grid_err:.3g} h", grid_err, GRID_TOL_H
            )
        ref = _reference(case.seed, name)
        err = float(np.max(np.abs(out["conc_mg_per_L"] - ref)) / ref.max())
        return Outcome(
            err <= CONC_TOL_REL, f"concentration error {err:.3g} of peak", err, CONC_TOL_REL
        )

    return FunctionCheck(code, fn)


SCENARIO_CODES = (
    ("standard", "STANDARD"),
    ("short_infusion", "SHORT_INFUSION"),
    ("bolus", "BOLUS"),
    ("long_horizon", "LONG_HORIZON"),
    ("coincident", "COINCIDENT_DOSES"),
)


@register
class PkDebugTask(TaskPlugin):
    task_id = "t05-pk-debug"
    version = "1.0.0"
    timeout_s = 60.0
    extra_env = (("PYTHONPATH", "{submission_dir}"),)

    def make_case(self, seed: int) -> DebugCase:
        return make_case(seed)

    def write_inputs(self, case: DebugCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("-m", "pkmodel.cli")

    def run_plans(self, case: Any) -> tuple[RunPlan, ...]:
        return tuple(
            RunPlan(
                name,
                (
                    "--params",
                    f"{{input_dir}}/{name}/params.json",
                    "--regimen",
                    f"{{input_dir}}/{name}/regimen.json",
                    "--out",
                    "{output_dir}/conc.csv",
                ),
            )
            for name, _ in SCENARIO_CODES
        )

    def checks(self) -> CheckGroup:
        return CheckGroup("t05", [scenario_check(n, c) for n, c in SCENARIO_CODES])
