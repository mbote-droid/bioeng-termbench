"""t03 acceptance thresholds, pinned at their boundaries with synthetic
simulation records: just inside passes, just outside fails. Added after
grader mutation testing showed that halving or doubling TIR_MIN, TAR250_MAX,
CORRECTION_LIMIT_MIN or SETTLE_P2P_MAX went unnoticed by the end-to-end tests
(no probe sits near those boundaries)."""

import numpy as np
import pytest

from bioengbench.checks.base import CheckContext
from bioengbench.core.scoring import Status
from bioengbench.tasks import t03_glucose as T

DT = 0.5  # trace resolution, min
# Contract values (instruction.md), deliberately not read from the grader.
DAY_MIN, CORRECTION_MIN, SETTLE_WINDOW_MIN = 1440.0, 360.0, 120.0


def _trace(duration: float, values) -> np.ndarray:
    t = np.arange(0.0, duration + DT / 2, DT)
    g = (
        np.broadcast_to(np.asarray(values, dtype=float), t.shape).copy()
        if np.ndim(values) == 0
        else values
    )
    return np.column_stack([t, g])


def _pump(duration: float, requested: float = 10.0) -> np.ndarray:
    t = np.arange(0.0, duration, T.SAMPLE_MIN)
    n = t.size
    return np.column_stack(
        [t, np.ones(n), np.full(n, 120.0), np.full(n, requested), np.full(n, requested)]
    )


def _ctx(tmp_path, day_g=None, cor_g=None, requested=10.0) -> CheckContext:
    sim = tmp_path / "sim"
    sim.mkdir(exist_ok=True)
    n_day = int(DAY_MIN / DT) + 1
    n_cor = int(CORRECTION_MIN / DT) + 1
    day = _trace(DAY_MIN, np.full(n_day, 120.0) if day_g is None else day_g)
    t_cor = np.arange(n_cor) * DT
    default_cor = np.where(t_cor < 60.0, 250.0 - t_cor * 2.0, 120.0)
    cor = _trace(CORRECTION_MIN, default_cor if cor_g is None else cor_g)
    T.write_sim(sim, "day", day, _pump(DAY_MIN, requested))
    T.write_sim(sim, "correction", cor, _pump(CORRECTION_MIN, requested))
    return CheckContext(output_dir=tmp_path, truth=None, run_dirs={"sim": sim})


def _status(check, ctx) -> Status:
    return check.evaluate(ctx)[0].status


def test_baseline_record_passes_every_check(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    for c in (
        T.actuator,
        T.hypoglycaemia,
        T.time_in_range,
        T.time_above_250,
        T.correction_time,
        T.settling,
    ):
        assert _status(c, ctx) is Status.PASS, c.code


@pytest.mark.parametrize(("low", "expected"), [(70.0, Status.PASS), (69.9, Status.FAIL)])
def test_hypo_boundary(tmp_path, low, expected) -> None:
    g = np.full(int(DAY_MIN / DT) + 1, 120.0)
    g[500] = low
    assert _status(T.hypoglycaemia, _ctx(tmp_path, day_g=g)) is expected


@pytest.mark.parametrize(("fraction_out", "expected"), [(0.30, Status.PASS), (0.31, Status.FAIL)])
def test_time_in_range_boundary(tmp_path, fraction_out, expected) -> None:
    n = int(DAY_MIN / DT) + 1
    g = np.full(n, 120.0)
    g[1 : 1 + round(fraction_out * (n - 1))] = 200.0  # above range, below 250
    assert _status(T.time_in_range, _ctx(tmp_path, day_g=g)) is expected


@pytest.mark.parametrize(("fraction_high", "expected"), [(0.05, Status.PASS), (0.06, Status.FAIL)])
def test_time_above_250_boundary(tmp_path, fraction_high, expected) -> None:
    n = int(DAY_MIN / DT) + 1
    g = np.full(n, 120.0)
    g[1 : 1 + round(fraction_high * (n - 1))] = 260.0
    assert _status(T.time_above_250, _ctx(tmp_path, day_g=g)) is expected


@pytest.mark.parametrize(("t_below", "expected"), [(120.0, Status.PASS), (120.5, Status.FAIL)])
def test_correction_time_boundary(tmp_path, t_below, expected) -> None:
    t = np.arange(int(CORRECTION_MIN / DT) + 1) * DT
    g = np.where(t < t_below, 200.0, 120.0)
    assert _status(T.correction_time, _ctx(tmp_path, cor_g=g)) is expected


@pytest.mark.parametrize(("p2p", "expected"), [(50.0, Status.PASS), (50.5, Status.FAIL)])
def test_settling_oscillation_boundary(tmp_path, p2p, expected) -> None:
    t = np.arange(int(CORRECTION_MIN / DT) + 1) * DT
    g = np.where(t < 60.0, 170.0, 120.0)
    late = t >= CORRECTION_MIN - SETTLE_WINDOW_MIN
    g[late] = 120.0 + 0.5 * p2p * np.sign(np.sin(2 * np.pi * t[late] / 40.0) + 1e-12)
    assert _status(T.settling, _ctx(tmp_path, cor_g=g)) is expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [(80.0, Status.PASS), (79.0, Status.FAIL), (160.0, Status.PASS), (161.0, Status.FAIL)],
)
def test_settling_level_boundary(tmp_path, level, expected) -> None:
    t = np.arange(int(CORRECTION_MIN / DT) + 1) * DT
    g = np.where(t < 60.0, 170.0, level)
    assert _status(T.settling, _ctx(tmp_path, cor_g=g)) is expected


def test_settling_window_is_the_last_two_hours(tmp_path) -> None:
    """A swing just before the window does not count; inside it does."""
    t = np.arange(int(CORRECTION_MIN / DT) + 1) * DT
    start = CORRECTION_MIN - SETTLE_WINDOW_MIN
    g = np.where(t < 60.0, 170.0, 120.0)
    g[(t > start - 30) & (t < start - 1)] = 175.0
    assert _status(T.settling, _ctx(tmp_path, cor_g=g)) is Status.PASS
    g[(t > start + 1) & (t < start + 30)] = 175.0
    assert _status(T.settling, _ctx(tmp_path, cor_g=g)) is Status.FAIL


@pytest.mark.parametrize(
    ("rate", "expected"), [(120.0, Status.PASS), (120.1, Status.FAIL), (-0.1, Status.FAIL)]
)
def test_actuator_boundary(tmp_path, rate, expected) -> None:
    assert _status(T.actuator, _ctx(tmp_path, requested=rate)) is expected
