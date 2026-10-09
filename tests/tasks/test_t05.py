"""t05 grader validation: a bug-isolation matrix.

The fixed package passes; the shipped package fails; and each defect,
injected alone into the fixed package, fails exactly the scenario designed
for it. So every scenario is necessary, and no defect hides behind another."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import T05

from bioengbench.core.runner import LocalRunner
from bioengbench.core.scoring import Status
from bioengbench.tasks import t05_pk_debug as T

PY = LocalRunner((sys.executable,))
FIXED = T05 / "solution/pkmodel"
SHIPPED = T05 / "environment/pkmodel"
ALL = {code for _, code in T.SCENARIO_CODES}


def _variant(tmp_path: Path, files: dict[str, str] | None = None, edits=()) -> Path:
    """Fixed package with some modules taken from the shipped (buggy) one,
    and optional text edits."""
    dest = tmp_path / "variant" / "pkmodel"
    shutil.copytree(FIXED, dest, ignore=shutil.ignore_patterns("__pycache__"))
    for name in files or {}:
        shutil.copyfile(SHIPPED / name, dest / name)
    for name, old, new in edits:
        text = (dest / name).read_text()
        assert old in text, old
        (dest / name).write_text(text.replace(old, new))
    return dest


def _failed(t05, pkg: Path, seed: int, workdir: Path) -> set[str]:
    case = t05.grade_case(pkg.resolve(), seed, PY, workdir)
    return {r.code for r in case.results if r.status is Status.FAIL}


ISOLATED = {
    "units": ({"units.py": ""}, (), ALL),
    "grid_float32": ({"grid.py": ""}, (), {"LONG_HORIZON"}),
    "doses_keyed_by_time": ({"regimen.py": ""}, (), {"COINCIDENT_DOSES"}),
    "bolus_skipped": (
        None,
        [
            (
                "solver.py",
                "        return sum(d.amount_mg for d in doses if d.is_bolus and d.start_h == t)",
                "        return 0.0",
            )
        ],
        {"BOLUS"},
    ),
    # The shipped solver: loose-tolerance RK45 over the whole horizon (and no boluses).
    "adaptive_solver": ({"solver.py": ""}, (), {"STANDARD", "SHORT_INFUSION", "BOLUS"}),
}


@pytest.mark.parametrize("bug", sorted(ISOLATED))
def test_each_defect_alone_fails_its_scenario(t05, t05_dev_seeds, tmp_path, bug) -> None:
    files, edits, expected = ISOLATED[bug]
    failed = _failed(t05, _variant(tmp_path, files, edits), t05_dev_seeds[0], tmp_path / "w")
    if bug in {"units", "adaptive_solver"}:
        assert expected <= failed, failed
    else:
        assert failed == expected, failed


def test_fixed_package_passes(t05, t05_dev_seeds, tmp_path) -> None:
    assert not _failed(t05, FIXED, t05_dev_seeds[0], tmp_path)


def test_shipped_package_fails_every_scenario(t05, t05_dev_seeds, tmp_path) -> None:
    assert _failed(t05, SHIPPED, t05_dev_seeds[0], tmp_path) == ALL


def test_shipped_package_differs_only_in_the_planted_modules() -> None:
    differ = {p.name for p in FIXED.glob("*.py") if p.read_text() != (SHIPPED / p.name).read_text()}
    assert differ == {"units.py", "grid.py", "regimen.py", "solver.py"}


def test_adaptive_solver_steps_over_the_short_infusion(t05, t05_dev_seeds, tmp_path) -> None:
    """The defect is real, not a tolerance artefact: with the shipped solver
    the late short infusion is (nearly) missing from the output."""
    failed = _failed(t05, _variant(tmp_path, {"solver.py": ""}), t05_dev_seeds[1], tmp_path / "w")
    assert "SHORT_INFUSION" in failed


def test_exact_reference_against_an_independent_ode_solution() -> None:
    from scipy.integrate import solve_ivp

    case = T.make_case(t05_seed := 12345)
    s = next(x for x in case.scenarios if x.name == "bolus")
    t = s.times()
    ref = T.exact_concentrations(s.params, s.doses, t)
    p = s.params
    cl, v1, q, v2 = p["CL_mL_per_min"] * 0.06, p["V1_L"], p["Q_L_per_h"], p["V2_L"]
    inf = s.doses[1]

    def rhs(tt, x):
        r = (
            inf.amount_mg / inf.duration_h
            if inf.start_h <= tt < inf.start_h + inf.duration_h
            else 0.0
        )
        return [-(cl + q) / v1 * x[0] + q / v2 * x[1] + r, q / v1 * x[0] - q / v2 * x[1]]

    x0 = [s.doses[0].amount_mg, 0.0]
    sol = solve_ivp(
        rhs, (0, t[-1]), x0, t_eval=t, method="LSODA", rtol=1e-11, atol=1e-11, max_step=0.01
    )
    np.testing.assert_allclose(ref, sol.y[0] / v1, rtol=1e-6, atol=1e-8)
    assert t05_seed


@pytest.mark.slow
def test_fixed_package_passes_all_dev_seeds(t05, t05_dev_seeds, tmp_path) -> None:
    verdict = t05.grade(FIXED.resolve(), t05_dev_seeds, PY, tmp_path)
    assert verdict.passed, verdict.to_json()
