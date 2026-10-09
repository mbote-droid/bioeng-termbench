"""t04 acceptance thresholds pinned at their boundaries with synthetic
outputs built from the grader's reference, using the contract's numbers
(instruction.md) rather than the grader's constants. Complements the mutant
tests in test_t04.py, which show each check fires but sit far from its
threshold."""

import json
import math

import pytest
from scipy.optimize import brentq

from bioengbench.checks.base import CheckContext
from bioengbench.core.scoring import Status
from bioengbench.tasks import t04_ablation as T

# Contract values.
ORDER_MIN = 1.6
TOL_PEAK_C, TOL_RADIUS_MM, TOL_ENERGY_REL = 0.05, 0.10, 0.002
CLOSURE_REL, TOL_TERM_REL = 0.001, 0.01
TOL_PLAN_MM, TOL_PLAN_C = 0.15, 0.15
MARGIN_MM, PEAK_LIMIT_C, EFFICIENCY = 5.0, 100.0, 1.10
SPACE = ((200, 1.0), (400, 1.0), (800, 1.0))
TIME = ((400, 2.0), (400, 1.0), (400, 0.5))
BASE = (400, 1.0)


@pytest.fixture(scope="module")
def case(t04_dev_seeds):
    return T.make_case(t04_dev_seeds[0])


@pytest.fixture(scope="module")
def ref(case):
    return T.reference(case.problem, case.test_power_w, case.test_heating_s, case.perfusion_nominal)


def _sim(ref, **overrides) -> dict:
    doc = {
        "t_center_peak_C": ref.peak_c,
        "energy_at_heating_end_J": ref.energy_ton_j,
        "ablation_radius_mm": ref.radius_mm,
        "energy_budget_J": dict(ref.budget),
    }
    for k, v in overrides.items():
        if k.startswith("budget_"):
            doc["energy_budget_J"][k[7:]] = v
        else:
            doc[k] = v
    return doc


def _ctx(tmp_path, case, ref, sims: dict | None = None, plan: dict | None = None) -> CheckContext:
    """Simulation outputs at every level (reference values unless overridden)
    and a plan; returns a context with the loading gates already evaluated."""
    sims = sims or {}
    run_dirs = {}
    for lv in T.LEVELS:
        d = tmp_path / T.level_name(lv)
        d.mkdir(parents=True, exist_ok=True)
        (d / "simulation.json").write_text(json.dumps(sims.get(lv, _sim(ref))))
        run_dirs[T.level_name(lv)] = d
    main = tmp_path / "main"
    main.mkdir(exist_ok=True)
    plan = plan or {
        "heating_time_s": case.test_heating_s,
        "power_W": case.test_power_w,
        "predicted_ablation_radius_mm": 0.0,
        "predicted_peak_C": 0.0,
    }
    (main / "plan.json").write_text(json.dumps(plan))
    ctx = CheckContext(output_dir=main, truth=case, run_dirs=run_dirs)
    for gate in (T.plan_format, T.sim_format):
        assert gate.evaluate(ctx)[0].status is Status.PASS
    return ctx


def _status(check, ctx) -> Status:
    return check.evaluate(ctx)[0].status


def _levels_with_order(ref, levels, order: float, rel: float = 1e-4) -> dict:
    """f1, f2 = reference at the middle level, f3 such that
    |f1 - f2| / |f2 - f3| = 2^order, for both convergence functionals."""
    out = {}
    for i, lv in enumerate(levels):
        scale = {0: 2.0**order, 1: 0.0, 2: -1.0}[i]
        out[lv] = _sim(
            ref,
            t_center_peak_C=ref.peak_c * (1 + rel * scale),
            energy_at_heating_end_J=ref.energy_ton_j * (1 + rel * scale),
        )
    return out


@pytest.mark.parametrize("which", ["space", "time"])
@pytest.mark.parametrize(
    ("order", "expected"), [(ORDER_MIN + 0.02, Status.PASS), (ORDER_MIN - 0.02, Status.FAIL)]
)
def test_convergence_order_boundary(tmp_path, case, ref, which, order, expected) -> None:
    levels = SPACE if which == "space" else TIME
    check = T.convergence_space if which == "space" else T.convergence_time
    ctx = _ctx(tmp_path, case, ref, sims=_levels_with_order(ref, levels, order))
    assert _status(check, ctx) is expected


@pytest.mark.parametrize(
    ("field", "delta", "expected"),
    [
        ("t_center_peak_C", 0.98 * TOL_PEAK_C, Status.PASS),
        ("t_center_peak_C", 1.02 * TOL_PEAK_C, Status.FAIL),
        ("ablation_radius_mm", 0.98 * TOL_RADIUS_MM, Status.PASS),
        ("ablation_radius_mm", 1.02 * TOL_RADIUS_MM, Status.FAIL),
    ],
)
def test_accuracy_absolute_boundaries(tmp_path, case, ref, field, delta, expected) -> None:
    value = {"t_center_peak_C": ref.peak_c, "ablation_radius_mm": ref.radius_mm}[field] + delta
    ctx = _ctx(tmp_path, case, ref, sims={BASE: _sim(ref, **{field: value})})
    assert _status(T.accuracy, ctx) is expected


@pytest.mark.parametrize(
    ("rel", "expected"),
    [(0.98 * TOL_ENERGY_REL, Status.PASS), (1.02 * TOL_ENERGY_REL, Status.FAIL)],
)
def test_accuracy_energy_boundary(tmp_path, case, ref, rel, expected) -> None:
    ctx = _ctx(
        tmp_path,
        case,
        ref,
        sims={BASE: _sim(ref, energy_at_heating_end_J=ref.energy_ton_j * (1 + rel))},
    )
    assert _status(T.accuracy, ctx) is expected


@pytest.mark.parametrize(
    ("rel", "expected"), [(0.9 * CLOSURE_REL, Status.PASS), (1.1 * CLOSURE_REL, Status.FAIL)]
)
def test_budget_closure_boundary(tmp_path, case, ref, rel, expected) -> None:
    """Shift heat from 'deposited' into nothing: the residual equals the shift."""
    dep = ref.budget["deposited"]
    closed = ref.budget["stored"] + ref.budget["perfusion"] + ref.budget["boundary"]
    sim = _sim(ref, budget_deposited=closed / (1 - rel))
    assert dep == pytest.approx(closed, rel=1e-6)
    ctx = _ctx(tmp_path, case, ref, sims={BASE: sim})
    assert _status(T.energy_budget, ctx) is expected


@pytest.mark.parametrize(
    ("rel", "expected"), [(0.9 * TOL_TERM_REL, Status.PASS), (1.1 * TOL_TERM_REL, Status.FAIL)]
)
def test_budget_term_boundary(tmp_path, case, ref, rel, expected) -> None:
    """Move heat from the smaller of storage and perfusion to the larger:
    closure holds, and the smaller term's relative error is exactly `rel`."""
    b = ref.budget
    small, large = sorted(("stored", "perfusion"), key=lambda k: b[k])
    shift = rel * b[small]
    sim = _sim(ref, **{f"budget_{small}": b[small] - shift, f"budget_{large}": b[large] + shift})
    ctx = _ctx(tmp_path, case, ref, sims={BASE: sim})
    assert _status(T.energy_budget, ctx) is expected


def _truth(case, power, perfusion):
    return T.reference(case.problem, power, case.test_heating_s, perfusion)


@pytest.mark.parametrize(
    ("d_mm", "d_c", "expected"),
    [
        (0.98 * TOL_PLAN_MM, 0.0, Status.PASS),
        (1.02 * TOL_PLAN_MM, 0.0, Status.FAIL),
        (0.0, 0.98 * TOL_PLAN_C, Status.PASS),
        (0.0, 1.02 * TOL_PLAN_C, Status.FAIL),
    ],
)
def test_plan_consistency_boundary(tmp_path, case, ref, d_mm, d_c, expected) -> None:
    hi = _truth(case, case.test_power_w, case.perfusion_hi)
    lo = _truth(case, case.test_power_w, case.perfusion_lo)
    plan = {
        "heating_time_s": case.test_heating_s,
        "power_W": case.test_power_w,
        "predicted_ablation_radius_mm": hi.radius_mm + d_mm,
        "predicted_peak_C": lo.peak_c + d_c,
    }
    assert _status(T.plan_consistent, _ctx(tmp_path, case, ref, plan=plan)) is expected


def _plan(case, power):
    return {
        "heating_time_s": case.test_heating_s,
        "power_W": power,
        "predicted_ablation_radius_mm": 0.0,
        "predicted_peak_C": 0.0,
    }


def test_coverage_boundary_uses_a_five_mm_margin(tmp_path, case, ref) -> None:
    need = case.tumour_radius_mm + MARGIN_MM
    p = brentq(
        lambda p: _truth(case, p, case.perfusion_hi).radius_mm - need,
        0.3 * case.test_power_w,
        case.test_power_w,
        xtol=1e-6,
    )
    assert (
        _status(T.coverage, _ctx(tmp_path / "a", case, ref, plan=_plan(case, p * 1.002)))
        is Status.PASS
    )
    assert (
        _status(T.coverage, _ctx(tmp_path / "b", case, ref, plan=_plan(case, p * 0.998)))
        is Status.FAIL
    )


def test_safety_boundary_is_one_hundred_degrees(tmp_path, case, ref) -> None:
    p = brentq(
        lambda p: _truth(case, p, case.perfusion_lo).peak_c - PEAK_LIMIT_C,
        case.test_power_w,
        3 * case.test_power_w,
        xtol=1e-6,
    )
    assert (
        _status(T.safety, _ctx(tmp_path / "a", case, ref, plan=_plan(case, p * 0.999)))
        is Status.PASS
    )
    assert (
        _status(T.safety, _ctx(tmp_path / "b", case, ref, plan=_plan(case, p * 1.001)))
        is Status.FAIL
    )


@pytest.mark.parametrize(
    ("ratio", "expected"), [(0.99 * EFFICIENCY, Status.PASS), (1.01 * EFFICIENCY, Status.FAIL)]
)
def test_efficiency_boundary(tmp_path, case, ref, ratio, expected) -> None:
    power = ratio * case.energy_min / case.test_heating_s
    ctx = _ctx(tmp_path, case, ref, plan=_plan(case, power))
    assert _status(T.efficiency, ctx) is expected


def test_contract_constants_match_the_instruction() -> None:
    assert math.isclose(T.MARGIN_MM, MARGIN_MM) and math.isclose(T.PEAK_LIMIT_C, PEAK_LIMIT_C)


@pytest.mark.parametrize(
    ("heating", "power", "expected"),
    [
        (180.0, 10.0, Status.PASS),
        (250.0, 10.0, Status.FAIL),  # not on the menu (contract: one of heating_times_s)
        (180.0, 0.0, Status.FAIL),
        (180.0, 200.0, Status.PASS),
        (180.0, 200.5, Status.FAIL),
    ],
)
def test_plan_format_contract(tmp_path, heating, power, expected) -> None:
    main = tmp_path / "main"
    main.mkdir()
    plan = {
        "heating_time_s": heating,
        "power_W": power,
        "predicted_ablation_radius_mm": 10.0,
        "predicted_peak_C": 90.0,
    }
    (main / "plan.json").write_text(json.dumps(plan))
    ctx = CheckContext(output_dir=main, truth=None, run_dirs={})
    assert _status(T.plan_format, ctx) is expected
