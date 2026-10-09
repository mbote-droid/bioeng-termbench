"""t04 grader validation. Three independent references pass (finite volumes,
finite elements, Octave); each realistic mistake fails with the right code.

Mistakes are injected as source edits of the reference (mutants), so each
control differs from a passing program in exactly one idea."""

import sys
from pathlib import Path

import pytest
from conftest import T04

from bioengbench.core.runner import LocalRunner
from bioengbench.core.scoring import Status
from bioengbench.tasks import t04_ablation as T

PY = LocalRunner((sys.executable,))
SOLUTION = T04 / "solution/ablation.py"

MUTANTS = {
    # Crank-Nicolson with perfusion lagged one step: first order in time.
    "lagged_perfusion": ([("for _ in range(3):", "for _ in range(1):")], {"CONVERGENCE_TIME"}),
    # Slab instead of sphere: wrong volumes and face areas.
    "cartesian": (
        [
            (
                "vol = 4 * np.pi / 3 * (f[1:] ** 3 - f[:-1] ** 3)",
                "vol = 4 * np.pi * sigma**2 * h * np.ones(nr)",
            ),
            (
                "g = k * 4 * np.pi * f[1:-1] ** 2 / h",
                "g = k * 4 * np.pi * sigma**2 * np.ones(nr - 1) / h",
            ),
        ],
        {"ACCURACY"},
    ),
    # Constant perfusion: ignores vascular collapse.
    "no_collapse": (
        [("return expit((stasis_c - t_body - th) / stasis_w)", "return np.ones_like(th)")],
        {"ACCURACY"},
    ),
    # Ignores --nr/--dt and always runs its own resolution.
    "ignores_resolution": (
        [
            (
                "res = simulate(cfg, a.power, a.heating_time, a.perfusion, a.nr, a.dt)",
                "res = simulate(cfg, a.power, a.heating_time, a.perfusion, 400, 1.0)",
            )
        ],
        {"CONVERGENCE_SPACE", "CONVERGENCE_TIME"},
    ),
    # Reports stored energy at the end of heating instead of the end of the run.
    "budget_wrong_time": ([('"stored": cap @ th', '"stored": e_on')], {"ENERGY_BUDGET"}),
    # Plans for the nominal perfusion only: not robust.
    "nominal_only": (
        [
            (
                'lo, hi = cfg["tissue"]["perfusion_range_per_s"]',
                'lo, hi = [sum(cfg["tissue"]["perfusion_range_per_s"]) / 2] * 2',
            )
        ],
        {"COVERAGE"},
    ),
    # Always heats as long as allowed (still safe and covering, but wasteful).
    "longest_time": ([("p * t < best[0] * best[1]", "t > best[1]")], {"EFFICIENCY"}),
    # Predicts at the nominal perfusion instead of the extremes.
    "inconsistent_prediction": (
        [
            (
                "at_hi = simulate(cfg, p, t, hi, 400, 1.0)",
                "at_hi = simulate(cfg, p, t, 0.5 * (lo + hi), 400, 1.0)",
            )
        ],
        {"PLAN_CONSISTENT"},
    ),
}


def _mutant(tmp_path: Path, edits) -> Path:
    src = SOLUTION.read_text()
    for old, new in edits:
        assert old in src, old
        src = src.replace(old, new)
    path = tmp_path / "ablation.py"
    path.write_text(src)
    return path


def _failed(t04, sub: Path, seed: int, workdir: Path) -> set[str]:
    case = t04.grade_case(sub.resolve(), seed, PY, workdir)
    return {r.code for r in case.results if r.status is Status.FAIL}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_fails_with_the_expected_code(t04, t04_dev_seeds, tmp_path, name) -> None:
    edits, expected = MUTANTS[name]
    failed = _failed(t04, _mutant(tmp_path, edits), t04_dev_seeds[0], tmp_path / "work")
    assert expected <= failed, failed


def test_inputs_hold_no_truth(t04, t04_dev_seeds, tmp_path) -> None:
    case = t04.make_case(t04_dev_seeds[0])
    t04.write_inputs(case, tmp_path)
    text = (tmp_path / "problem.json").read_text()
    assert "p_min" not in text and "window" not in text
    assert repr(case.test_power_w) not in text


def test_order_is_undefined_when_results_ignore_resolution() -> None:
    assert T._order([1.0, 1.0, 1.0]) != T._order([1.0, 1.0, 1.0])  # NaN


def test_case_has_a_robust_plan_with_room_for_margins(t04, t04_dev_seeds) -> None:
    for seed in t04_dev_seeds[:2]:
        case = t04.make_case(seed)
        best = min((w for w in case.windows if w.feasible), key=lambda w: w.energy_min)
        assert best.p_max / best.p_min - 1 >= T.MIN_WINDOW
        assert case.energy_min == pytest.approx(best.energy_min)


@pytest.mark.slow
@pytest.mark.parametrize("ref", ["solution/ablation.py", "references/ablation_fem.py"])
def test_python_references_pass_dev_seeds(t04, t04_dev_seeds, tmp_path, ref) -> None:
    verdict = t04.grade((T04 / ref).resolve(), t04_dev_seeds[:4], PY, tmp_path)
    assert verdict.passed, verdict.to_json()


@pytest.mark.slow
@pytest.mark.octave
def test_octave_reference_passes(t04, t04_dev_seeds, octave, tmp_path) -> None:
    runner = LocalRunner((octave, "--no-gui", "--quiet"))
    verdict = t04.grade(
        (T04 / "references/ablation.m").resolve(), t04_dev_seeds[:2], runner, tmp_path, "octave"
    )
    assert verdict.passed, verdict.to_json()
