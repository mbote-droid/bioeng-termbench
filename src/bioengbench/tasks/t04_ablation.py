"""t04: thermal ablation planning with a verified bioheat solver.

A tumour of radius R_t must be destroyed with a 5 mm margin by a probe at its
centre that deposits a Gaussian heat source. The agent writes one program with
two modes:

  simulate  solve Pennes bioheat (perfusion collapsing with temperature) and
            Arrhenius damage for a given power, heating time, perfusion and
            numerical resolution
  plan      choose the heating time (from a fixed menu) and the power

The grader reruns `simulate` at refined resolutions, in space (h, h/2, h/4)
and in time (dt, dt/2, dt/4) separately, and measures the observed order of
accuracy from the differences between levels: it must be about 2 in both
(Crank-Nicolson in time, second order in space). A first-order method,
including Crank-Nicolson with the temperature-dependent perfusion lagged by
one step, fails. The base run (400 cells, 1 s) must agree with the grader's
Richardson-extrapolated reference, and the reported energy budget must close.

The plan is judged at the truth with the grader's reference solver. Blood
perfusion is uncertain within +-20 %, so the plan must be robust: the lesion
must reach R_t + 5 mm at the highest perfusion (most heat carried away), and
the centre must stay at or below 100 C (boiling, charring) at the lowest. Of
the plans that meet both, the delivered energy must be within 10 % of the
smallest possible. The decision is driven by the physics: short, intense
heating wastes less energy to perfusion but runs into the temperature limit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from scipy.optimize import brentq

from bioengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from bioengbench.checks.io import SchemaError, read_json_numbers
from bioengbench.core.registry import register
from bioengbench.core.seeds import streams
from bioengbench.core.task import MAIN, RunPlan, TaskPlugin
from bioengbench.models.bioheat import (
    ARRHENIUS_A,
    ARRHENIUS_EA,
    R_GAS,
    STASIS_C,
    STASIS_WIDTH_K,
    T_BODY_C,
    Problem,
    Result,
    simulate,
)

HEATING_TIMES = (180.0, 300.0, 420.0, 600.0, 900.0, 1200.0)  # s
COOLING_S = 300.0  # damage keeps accumulating after the power is switched off
MARGIN_MM = 5.0
PEAK_LIMIT_C = 100.0
PERFUSION_SPREAD = 0.20  # +-20 % around the nominal value
EFFICIENCY_LIMIT = 1.10
MIN_WINDOW = 0.05  # feasible power window at the optimum, relative; leaves room for margins

# Convergence study (cells, time step in s). Space and time are refined
# separately: refining both together lets the spatial and temporal error
# terms partly cancel, and the observed order then scatters (one reference
# case in 60 measured 0.76). Refining one at a time removes the other's
# error from the differences, so each order is clean (measured 1.98-2.01 for
# second-order schemes, 0.98-1.00 for lagged perfusion in time). The order is
# measured on the stored energy (see _convergence).
BASE = (400, 1.0)
SPACE_LEVELS = ((200, 1.0), (400, 1.0), (800, 1.0))
TIME_LEVELS = ((400, 2.0), (400, 1.0), (400, 0.5))
LEVELS = tuple(dict.fromkeys(SPACE_LEVELS + TIME_LEVELS))  # distinct runs
ORDER_MIN = 1.6
SEARCH_LEVEL = (200, 2.0)  # grader's plan search; error << tolerances below
TRUTH_LEVELS = ((200, 2.0), (400, 1.0))  # Richardson pair for the reference
TOL_PEAK_C = 0.05
TOL_ENERGY_REL = 0.002
TOL_RADIUS_MM = 0.10
TOL_BUDGET_REL = 0.01
CLOSURE_REL = 1e-3
TOL_PLAN_RADIUS_MM = 0.15
TOL_PLAN_PEAK_C = 0.15

PLAN_KEYS = ("heating_time_s", "power_W", "predicted_ablation_radius_mm", "predicted_peak_C")
SIM_KEYS = ("t_center_peak_C", "energy_at_heating_end_J", "ablation_radius_mm")
BUDGET_KEYS = ("deposited", "stored", "perfusion", "boundary")


@dataclass(frozen=True)
class Window:
    heating_s: float
    p_min: float  # coverage at the highest perfusion
    p_max: float  # peak limit at the lowest perfusion

    @property
    def feasible(self) -> bool:
        return self.p_min <= self.p_max

    @property
    def energy_min(self) -> float:
        return self.p_min * self.heating_s


@dataclass(frozen=True)
class Reference:
    """Richardson-extrapolated reference values for one simulation."""

    peak_c: float
    energy_ton_j: float
    radius_mm: float
    budget: dict[str, float]


@dataclass(frozen=True)
class AblationCase:
    seed: int
    problem: Problem
    tumour_radius_mm: float
    perfusion_nominal: float
    windows: tuple[Window, ...]
    test_power_w: float  # convergence-study inputs
    test_heating_s: float

    @property
    def perfusion_lo(self) -> float:
        return self.perfusion_nominal * (1.0 - PERFUSION_SPREAD)

    @property
    def perfusion_hi(self) -> float:
        return self.perfusion_nominal * (1.0 + PERFUSION_SPREAD)

    @property
    def required_radius_mm(self) -> float:
        return self.tumour_radius_mm + MARGIN_MM

    @property
    def energy_min(self) -> float:
        return min(w.energy_min for w in self.windows if w.feasible)


def run(
    problem: Problem, power: float, heating: float, perfusion: float, level: tuple[int, float]
) -> Result:
    nr, dt = level
    return simulate(
        problem,
        power_w=power,
        heating_s=heating,
        perfusion_per_s=perfusion,
        end_s=heating + COOLING_S,
        nr=nr,
        dt=dt,
    )


@lru_cache(maxsize=256)
def reference(problem: Problem, power: float, heating: float, perfusion: float) -> Reference:
    """Second-order Richardson extrapolation from the (h, dt) and (h/2, dt/2) runs."""
    coarse, fine = (run(problem, power, heating, perfusion, lv) for lv in TRUTH_LEVELS)

    def rich(a: float, b: float) -> float:
        return b + (b - a) / 3.0

    budget = {k: rich(getattr(coarse.budget, k), getattr(fine.budget, k)) for k in BUDGET_KEYS}
    return Reference(
        rich(coarse.t_center_peak_c, fine.t_center_peak_c),
        rich(coarse.energy_at_ton_j, fine.energy_at_ton_j),
        rich(coarse.ablation_radius_mm, fine.ablation_radius_mm),
        budget,
    )


def power_window(
    problem: Problem, heating: float, lo: float, hi: float, required_mm: float
) -> Window:
    def peak(p: float) -> float:
        return run(problem, p, heating, lo, SEARCH_LEVEL).t_center_peak_c - PEAK_LIMIT_C

    def cover(p: float) -> float:
        return run(problem, p, heating, hi, SEARCH_LEVEL).ablation_radius_mm - required_mm

    p_max = brentq(peak, 0.1, 200.0, xtol=1e-4, rtol=1e-6)
    if cover(p_max) < 0:
        return Window(heating, math.inf, p_max)
    p_min = brentq(cover, 0.1, p_max, xtol=1e-4, rtol=1e-6)
    return Window(heating, p_min, p_max)


@lru_cache(maxsize=64)
def make_case(seed: int) -> AblationCase:
    (rng,) = streams(seed, 1)
    while True:  # rejection sampling: only cases with a robust plan that leaves room for margins
        problem = Problem(sigma_m=rng.uniform(0.005, 0.009))
        r_t = rng.uniform(4.0, 8.0)
        nominal = rng.uniform(0.001, 0.006)
        lo, hi = nominal * (1 - PERFUSION_SPREAD), nominal * (1 + PERFUSION_SPREAD)
        windows = tuple(power_window(problem, t, lo, hi, r_t + MARGIN_MM) for t in HEATING_TIMES)
        feasible = [w for w in windows if w.feasible]
        if not feasible:
            continue
        best = min(feasible, key=lambda w: w.energy_min)
        if best.p_max / best.p_min - 1.0 < MIN_WINDOW:
            continue
        return AblationCase(
            seed, problem, r_t, nominal, windows, 0.5 * (best.p_min + best.p_max), best.heating_s
        )


def problem_json(case: AblationCase) -> dict[str, Any]:
    tis = case.problem.tissue
    return {
        "tissue": {
            "conductivity_W_per_m_K": tis.conductivity,
            "density_kg_per_m3": tis.density,
            "specific_heat_J_per_kg_K": tis.specific_heat,
            "blood_volumetric_heat_capacity_J_per_m3_K": tis.blood_heat_capacity,
            "perfusion_range_per_s": [case.perfusion_lo, case.perfusion_hi],
            "stasis_temperature_C": STASIS_C,
            "stasis_width_K": STASIS_WIDTH_K,
        },
        "arrhenius": {"A_per_s": ARRHENIUS_A, "Ea_J_per_mol": ARRHENIUS_EA, "R_J_per_mol_K": R_GAS},
        "body_temperature_C": T_BODY_C,
        "domain_radius_m": case.problem.domain_radius_m,
        "source_sigma_m": case.problem.sigma_m,
        "tumour_radius_mm": case.tumour_radius_mm,
        "margin_mm": MARGIN_MM,
        "peak_limit_C": PEAK_LIMIT_C,
        "heating_times_s": list(HEATING_TIMES),
        "cooling_time_s": COOLING_S,
    }


def write_inputs(case: AblationCase, input_dir: Path) -> None:
    (input_dir / "problem.json").write_text(json.dumps(problem_json(case), indent=2) + "\n")


def level_name(level: tuple[int, float]) -> str:
    return f"nr{level[0]}-dt{level[1]:g}"


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> AblationCase:
    case: AblationCase = ctx.truth
    return case


def _read_sim(path: Path) -> dict[str, float]:
    values = read_json_numbers(path, SIM_KEYS)
    raw = json.loads(path.read_text())
    budget = raw.get("energy_budget_J")
    if not isinstance(budget, dict):
        raise SchemaError(
            f"{path.parent.name}/simulation.json: 'energy_budget_J' must be an object"
        )
    for key in BUDGET_KEYS:
        v = budget.get(key)
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise SchemaError(f"energy_budget_J.{key} must be a finite number")
        values[f"budget_{key}"] = float(v)
    return values


@check("PLAN_FORMAT", gate=True)
def plan_format(ctx: CheckContext) -> Outcome:
    plan = read_json_numbers(ctx.output_dir / "plan.json", PLAN_KEYS)
    if plan["heating_time_s"] not in HEATING_TIMES:
        return Outcome(False, f"heating_time_s must be one of {HEATING_TIMES}")
    if not 0.0 < plan["power_W"] <= 200.0:
        return Outcome(False, "power_W must be in (0, 200]")
    ctx.outputs["plan"] = plan
    return Outcome(True)


@check("SIM_FORMAT", gate=True)
def sim_format(ctx: CheckContext) -> Outcome:
    ctx.outputs["levels"] = {
        lv: _read_sim(ctx.run_dirs[level_name(lv)] / "simulation.json") for lv in LEVELS
    }
    return Outcome(True)


def _order(values: list[float]) -> float:
    d1, d2 = abs(values[0] - values[1]), abs(values[1] - values[2])
    scale = max(abs(v) for v in values)
    if d1 <= 1e-12 * scale or d2 <= 1e-12 * scale:
        return math.nan  # results do not depend on the resolution
    return math.log2(d1 / d2)


def _convergence(ctx: CheckContext, levels: tuple[tuple[int, float], ...], what: str) -> Outcome:
    """Observed order p = log2(|E_1 - E_2| / |E_2 - E_3|) of the stored energy
    along one refinement path; it must be second order.

    The stored energy is used, not the peak temperature: a volume integral
    converges smoothly for every correct scheme (measured 1.99-2.11 for finite
    volumes and P1 finite elements), whereas a point value at the centre can
    carry two error terms of opposite sign. The P1 reference's peak measured
    orders of 0.36 and 1.49 on two of eight cases on these grids while its
    energy measured 2.0. The peak is still checked, for accuracy."""
    runs = [ctx.outputs["levels"][lv]["energy_at_heating_end_J"] for lv in levels]
    order = _order(runs)
    if math.isnan(order):
        flag = "--nr" if what == "space" else "--dt"
        return Outcome(
            False, f"results do not change with {flag}: the given resolution must be used"
        )
    return Outcome(order >= ORDER_MIN, f"observed order in {what} {order:.2f}", order, ORDER_MIN)


@check("CONVERGENCE_SPACE")
def convergence_space(ctx: CheckContext) -> Outcome:
    return _convergence(ctx, SPACE_LEVELS, "space")


@check("CONVERGENCE_TIME")
def convergence_time(ctx: CheckContext) -> Outcome:
    return _convergence(ctx, TIME_LEVELS, "time")


@check("ACCURACY")
def accuracy(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    ref = reference(case.problem, case.test_power_w, case.test_heating_s, case.perfusion_nominal)
    fine = ctx.outputs["levels"][BASE]
    errors = [
        ("peak temperature", abs(fine["t_center_peak_C"] - ref.peak_c), TOL_PEAK_C, "C"),
        ("ablation radius", abs(fine["ablation_radius_mm"] - ref.radius_mm), TOL_RADIUS_MM, "mm"),
        (
            "stored energy",
            abs(fine["energy_at_heating_end_J"] / ref.energy_ton_j - 1.0),
            TOL_ENERGY_REL,
            "relative",
        ),
    ]
    for name, err, tol, unit in errors:
        if err > tol:
            return Outcome(False, f"{name} off by {err:.3g} {unit} (tolerance {tol:g})", err, tol)
    return Outcome(True)


@check("ENERGY_BUDGET")
def energy_budget(ctx: CheckContext) -> Outcome:
    """The budget must close (conservation) and each term must be right."""
    case = _case(ctx)
    fine = ctx.outputs["levels"][BASE]
    b = {k: fine[f"budget_{k}"] for k in BUDGET_KEYS}
    residual = abs(b["deposited"] - b["stored"] - b["perfusion"] - b["boundary"]) / b["deposited"]
    if residual > CLOSURE_REL:
        return Outcome(False, "energy budget does not close", residual, CLOSURE_REL)
    ref = reference(case.problem, case.test_power_w, case.test_heating_s, case.perfusion_nominal)
    for key in ("deposited", "stored", "perfusion"):
        err = abs(b[key] / ref.budget[key] - 1.0)
        if err > TOL_BUDGET_REL:
            return Outcome(False, f"budget term '{key}' off by {err:.2%}", err, TOL_BUDGET_REL)
    # The boundary term is tiny (far field); compare it on the deposited scale.
    err = abs(b["boundary"] - ref.budget["boundary"]) / ref.budget["deposited"]
    return Outcome(err <= TOL_BUDGET_REL, "budget term 'boundary' wrong", err, TOL_BUDGET_REL)


def _plan_truth(case: AblationCase, plan: dict[str, float]) -> tuple[Reference, Reference]:
    p, t = plan["power_W"], plan["heating_time_s"]
    return (
        reference(case.problem, p, t, case.perfusion_hi),
        reference(case.problem, p, t, case.perfusion_lo),
    )


@check("PLAN_CONSISTENT")
def plan_consistent(ctx: CheckContext) -> Outcome:
    """The predictions must come from a correct simulation of the plan."""
    plan = ctx.outputs["plan"]
    hi, lo = _plan_truth(_case(ctx), plan)
    err_r = abs(plan["predicted_ablation_radius_mm"] - hi.radius_mm)
    if err_r > TOL_PLAN_RADIUS_MM:
        return Outcome(False, f"predicted radius off by {err_r:.3f} mm", err_r, TOL_PLAN_RADIUS_MM)
    err_t = abs(plan["predicted_peak_C"] - lo.peak_c)
    return Outcome(
        err_t <= TOL_PLAN_PEAK_C, f"predicted peak off by {err_t:.3f} C", err_t, TOL_PLAN_PEAK_C
    )


@check("COVERAGE")
def coverage(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    hi, _ = _plan_truth(case, ctx.outputs["plan"])
    return Outcome(
        hi.radius_mm >= case.required_radius_mm,
        "lesion does not cover tumour + margin at the highest perfusion",
        hi.radius_mm,
        case.required_radius_mm,
    )


@check("SAFETY")
def safety(ctx: CheckContext) -> Outcome:
    _, lo = _plan_truth(_case(ctx), ctx.outputs["plan"])
    return Outcome(
        lo.peak_c <= PEAK_LIMIT_C,
        "centre exceeds the temperature limit at the lowest perfusion",
        lo.peak_c,
        PEAK_LIMIT_C,
    )


@check("EFFICIENCY")
def efficiency(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    plan = ctx.outputs["plan"]
    ratio = plan["power_W"] * plan["heating_time_s"] / case.energy_min
    return Outcome(
        ratio <= EFFICIENCY_LIMIT, "delivered energy far above the minimum", ratio, EFFICIENCY_LIMIT
    )


@register
class AblationTask(TaskPlugin):
    task_id = "t04-bioheat-ablation"
    version = "1.0.0"
    timeout_s = 600.0

    def languages(self) -> tuple[str, ...]:
        return ("python", "octave")

    def make_case(self, seed: int) -> AblationCase:
        return make_case(seed)

    def write_inputs(self, case: AblationCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}",)

    def run_plans(self, case: AblationCase) -> tuple[RunPlan, ...]:
        problem = ("--problem", "{input_dir}/problem.json", "--outdir", "{output_dir}")
        plans = [RunPlan(MAIN, ("plan", *problem))]
        for nr, dt in LEVELS:
            plans.append(
                RunPlan(
                    level_name((nr, dt)),
                    (
                        "simulate",
                        *problem,
                        "--power",
                        repr(case.test_power_w),
                        "--heating-time",
                        repr(case.test_heating_s),
                        "--perfusion",
                        repr(case.perfusion_nominal),
                        "--nr",
                        str(nr),
                        "--dt",
                        repr(dt),
                    ),
                )
            )
        return tuple(plans)

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t04",
            [
                CheckGroup("load", [plan_format, sim_format], gate=True),
                CheckGroup(
                    "solver", [convergence_space, convergence_time, accuracy, energy_budget]
                ),
                CheckGroup("plan", [plan_consistent, coverage, safety, efficiency]),
            ],
        )
