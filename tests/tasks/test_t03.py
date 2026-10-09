"""t03 grader validation: the reference controllers pass; unsafe, sluggish,
oscillating and protocol-breaking controllers fail with the right code."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import T03

from bioengbench.core.runner import LocalRunner
from bioengbench.core.scoring import Status
from bioengbench.tasks import t03_glucose as T

CONTROLLERS = Path(__file__).parent / "t03_controllers"
SEED = 0x3D1F_06A2_5B7C_9E48_C1D0_2F63_A7B5_8E19
PY = LocalRunner((sys.executable,))


def _grade(t03, sub: Path, tmp_path, seed=SEED):
    case = t03.grade_case(sub.resolve(), seed, PY, tmp_path)
    return {r.code: r for r in case.results}


def _failed(results) -> set[str]:
    return {code for code, r in results.items() if r.status is Status.FAIL}


def test_reference_passes_one_case(t03, tmp_path) -> None:
    results = _grade(t03, T03 / "solution/controller.py", tmp_path)
    assert not _failed(results), results
    assert results["HYPO"].measured > 75.0


def test_inputs_reveal_nothing_about_the_patient(t03, tmp_path) -> None:
    case = t03.make_case(SEED)
    t03.write_inputs(case, tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["problem.json"]
    problem = json.loads((tmp_path / "problem.json").read_text())
    assert set(problem) == {
        "sample_period_min",
        "basal_rate_mU_per_min",
        "pump_max_mU_per_min",
        "units",
    }


def test_simulation_files_are_written_outside_the_controller_directory(t03, tmp_path) -> None:
    _grade(t03, T03 / "solution/controller.py", tmp_path)
    case_dir = next(tmp_path.glob("case-*"))
    assert sorted(p.name for p in (case_dir / "sim").iterdir()) == [
        "correction_pump.csv",
        "correction_trace.csv",
        "day_pump.csv",
        "day_trace.csv",
    ]
    assert not any((case_dir / "controller-day").glob("*.csv"))


@pytest.mark.parametrize(
    ("controller", "expected"),
    [
        ("basal_only", {"TIME_IN_RANGE", "TIME_ABOVE_250", "CORRECTION_TIME"}),
        ("aggressive_p", {"HYPO"}),
        ("integral_only", {"HYPO", "SETTLING"}),
        ("over_max", {"ACTUATOR"}),
    ],
)
def test_unsafe_or_poor_controllers_fail(
    t03, t03_dev_seeds, tmp_path, controller, expected
) -> None:
    """Each control is graded on three dev patients; its expected failure codes
    must all appear. (A single patient can be benign: aggressive_p fails HYPO on
    29 of 30 random patients, not all.)"""
    failed: set[str] = set()
    for seed in t03_dev_seeds[:3]:
        failed |= _failed(_grade(t03, CONTROLLERS / f"{controller}.py", tmp_path, seed))
    assert expected <= failed, failed


@pytest.mark.parametrize(
    ("controller", "reason"),
    [
        ("crash_on_dropout", "exited"),
        ("garbage", "JSON"),
        ("string_rate", "finite number"),
        ("slow", "deadline"),
    ],
)
def test_protocol_violations_fail_run(t03, tmp_path, controller, reason) -> None:
    results = _grade(t03, CONTROLLERS / f"{controller}.py", tmp_path)
    assert results["RUN"].status is Status.FAIL
    assert reason in results["RUN"].message, results["RUN"].message
    assert all(r.status is Status.SKIPPED for c, r in results.items() if c != "RUN")


@pytest.mark.slow
def test_naive_pid_fails_on_some_dev_seed(t03, t03_dev_seeds, tmp_path) -> None:
    """A plausible PID without safety layers is unsafe on a sizeable fraction
    of patients (measured: ~40 % of cases), so the all-cases reward rejects it."""
    verdict = t03.grade((CONTROLLERS / "naive_pid.py").resolve(), t03_dev_seeds, PY, tmp_path)
    assert not verdict.passed


def test_time_fraction_excludes_the_initial_sample() -> None:
    assert T._time_fraction(np.array([False, True, True, True])) == 1.0


def test_pump_saturation_is_applied(t03, tmp_path) -> None:
    _grade(t03, CONTROLLERS / "over_max.py", tmp_path)
    pump = np.genfromtxt(next(tmp_path.glob("case-*/sim/day_pump.csv")), delimiter=",", names=True)
    assert pump["requested_mU_per_min"].max() > T.U_MAX
    assert pump["delivered_mU_per_min"].max() == T.U_MAX


@pytest.mark.slow
def test_python_reference_passes_dev_seeds(t03, t03_dev_seeds, tmp_path) -> None:
    verdict = t03.grade((T03 / "solution/controller.py").resolve(), t03_dev_seeds, PY, tmp_path)
    assert verdict.passed, verdict.to_json()


@pytest.mark.slow
def test_verdict_is_byte_identical_across_runs(t03, t03_dev_seeds, tmp_path) -> None:
    sub = (T03 / "solution/controller.py").resolve()
    runs = [t03.grade(sub, t03_dev_seeds[:2], PY, tmp_path / str(i)) for i in range(2)]
    assert runs[0].to_json() == runs[1].to_json()


@pytest.mark.slow
@pytest.mark.octave
def test_octave_reference_passes_dev_seeds(t03, t03_dev_seeds, octave, tmp_path) -> None:
    runner = LocalRunner((octave, "--no-gui", "--quiet"))
    sub = (T03 / "references/controller.m").resolve()
    verdict = t03.grade(sub, t03_dev_seeds[:3], runner, tmp_path, "octave")
    assert verdict.passed, verdict.to_json()
    assert shutil.which("octave-cli")


def test_agent_test_bench_uses_the_grader_model() -> None:
    """environment/data/simulate.py is a standalone copy for the agent; it must
    not drift from the grader's plant."""
    import importlib.util

    from bioengbench.models.glucose import Disturbances, Exercise, Meal, Patient, derivative

    spec = importlib.util.spec_from_file_location("bench", T03 / "environment/data/simulate.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = bench.sample(rng)
        patient = Patient(p["mass"], p["p1"], p["p2"], p["p3"], p["n"], p["gb"], p["ub"], p["tau"])
        meals = [(rng.uniform(0, 100), rng.uniform(30, 60), rng.uniform(40, 60))]
        ex = (rng.uniform(0, 100), 45.0, rng.uniform(0.004, 0.008), rng.uniform(0.5, 1))
        d = Disturbances(tuple(Meal(*m) for m in meals), (Exercise(*ex),))
        x = np.array(
            [rng.uniform(60, 300), rng.normal(0, 0.01), rng.uniform(5, 40), rng.uniform(60, 300)]
        )
        t, u = rng.uniform(0, 300), rng.uniform(0, 120)
        np.testing.assert_allclose(
            bench.deriv(p, x, t, u, meals, ex), derivative(patient, x, t, u, d), rtol=1e-12
        )
