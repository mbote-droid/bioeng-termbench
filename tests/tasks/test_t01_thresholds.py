"""t01 acceptance thresholds pinned at their boundaries. Grader mutation
testing showed that halving or doubling several t01 tolerances went
unnoticed (the negative controls fail by wide margins). Each test builds an
output from ground truth that sits just inside, then just outside, one
threshold, and checks only that threshold's code."""

import json

import numpy as np
import pytest
from conftest import write_t01_outputs
from scipy.optimize import brentq

from bioengbench.checks.signal import band_power, bandlimited_rmse
from bioengbench.core.scoring import Status
from bioengbench.tasks import t01_ecg as T

# Contract values (instruction.md), deliberately not read from the grader, so a
# changed constant fails here. The last two are grader-internal and pinned too.
WANDER_RATIO = 0.10  # 10 dB
MAINS_RATIO = 0.05  # 13 dB
RMSE_MV = 0.05
AMP_TOL = 0.15
WINDOW_SAMPLES = 14  # 40 ms at 360 Hz
NBEATS_TOL = 2
TIME_INDEX_TOL_S = 1e-4
CONSISTENCY_MS = 1.0

SEED = 0x7A01_C3E5_0F12_4B6D_9E80_21A4_C6E8_0B13


@pytest.fixture(scope="module")
def case(t01):
    return t01.make_case(SEED)


def _status(t01, out, case, code) -> Status:
    return next(r.status for r in t01.evaluate_outputs(out, case) if r.code == code)


def _scaled_to(metric, target: float, kmax: float = 50.0) -> float:
    """k >= 0 with metric(k) = target (metric increasing in k)."""
    return brentq(lambda k: metric(k) - target, 0.0, kmax, xtol=1e-12)


@pytest.mark.parametrize(("side", "expected"), [(0.98, Status.PASS), (1.02, Status.FAIL)])
def test_baseline_ratio_boundary(t01, case, tmp_path, side, expected) -> None:
    d = case.raw - case.clean

    def ratio(k):
        y = case.clean + k * d
        return band_power(y, case.fs, T.BASELINE_BAND) / band_power(
            case.raw, case.fs, T.BASELINE_BAND
        )

    k = _scaled_to(ratio, side * WANDER_RATIO, 1.0)
    out = write_t01_outputs(tmp_path, case, filtered=case.clean + k * d)
    assert _status(t01, out, case, "BASELINE") is expected


@pytest.mark.parametrize(("side", "expected"), [(0.98, Status.PASS), (1.02, Status.FAIL)])
def test_powerline_ratio_boundary(t01, case, tmp_path, side, expected) -> None:
    d = case.raw - case.clean

    def ratio(k):
        y = case.clean + k * d
        return band_power(y, case.fs, T.LINE_BAND) / band_power(case.raw, case.fs, T.LINE_BAND)

    k = _scaled_to(ratio, side * MAINS_RATIO, 1.0)
    out = write_t01_outputs(tmp_path, case, filtered=case.clean + k * d)
    assert _status(t01, out, case, "POWERLINE") is expected


@pytest.mark.parametrize(("side", "expected"), [(0.98, Status.PASS), (1.02, Status.FAIL)])
def test_fidelity_boundary(t01, case, tmp_path, side, expected) -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(size=case.clean.size)
    trim = int(T.EDGE_S * case.fs)

    def rmse(k):
        return bandlimited_rmse(case.clean + k * d, case.clean, case.fs, T.MORPH_BAND, trim)

    k = _scaled_to(rmse, side * RMSE_MV, 1.0)
    out = write_t01_outputs(tmp_path, case, filtered=case.clean + k * d)
    assert _status(t01, out, case, "FIDELITY") is expected


@pytest.mark.parametrize(
    ("gain", "expected"),
    [
        (1.0 + 0.99 * AMP_TOL, Status.PASS),
        (1.0 + 1.01 * AMP_TOL, Status.FAIL),
        (1.0 - 1.01 * AMP_TOL, Status.FAIL),
    ],
)
def test_amplitude_boundary(t01, case, tmp_path, gain, expected) -> None:
    out = write_t01_outputs(tmp_path, case, filtered=case.clean * gain)
    assert _status(t01, out, case, "AMPLITUDE") is expected


@pytest.mark.parametrize(
    ("shift", "expected"), [(WINDOW_SAMPLES, Status.PASS), (WINDOW_SAMPLES + 1, Status.FAIL)]
)
def test_peak_window_boundary(t01, case, tmp_path, shift, expected) -> None:
    """40 ms at 360 Hz rounds to 14 samples."""
    peaks = case.r_idx + shift
    peaks = peaks[peaks < case.raw.size]
    out = write_t01_outputs(tmp_path, case, peaks=peaks, metrics=T.hrv(peaks, case.fs))
    assert _status(t01, out, case, "PEAK_SENS") is expected


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(0.9 * TIME_INDEX_TOL_S, Status.PASS), (1.1 * TIME_INDEX_TOL_S, Status.FAIL)],
)
def test_time_index_boundary(t01, case, tmp_path, offset, expected) -> None:
    out = write_t01_outputs(tmp_path, case)
    p = case.r_idx
    np.savetxt(
        out / "rpeaks.csv",
        np.column_stack([p, p / case.fs + offset]),
        delimiter=",",
        header="sample_index,time_s",
        comments="",
        fmt=["%d", "%.9f"],
    )
    assert _status(t01, out, case, "PEAK_FORMAT") is expected


@pytest.mark.parametrize(
    ("extra", "expected"), [(NBEATS_TOL, Status.PASS), (NBEATS_TOL + 1, Status.FAIL)]
)
def test_beat_count_boundary(t01, case, tmp_path, extra, expected) -> None:
    m = T.hrv(case.r_idx, case.fs) | {"n_beats": float(len(case.r_idx) + extra)}
    out = write_t01_outputs(tmp_path, case, metrics=m)
    assert _status(t01, out, case, "NBEATS") is expected


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.9 * CONSISTENCY_MS, Status.PASS), (1.1 * CONSISTENCY_MS, Status.FAIL)],
)
def test_hrv_consistency_boundary(t01, case, tmp_path, delta, expected) -> None:
    m = T.hrv(case.r_idx, case.fs)
    m["rmssd_ms"] += delta
    out = write_t01_outputs(tmp_path, case, metrics=m)
    assert json.loads((out / "metrics.json").read_text())["rmssd_ms"] == pytest.approx(
        m["rmssd_ms"]
    )
    assert _status(t01, out, case, "HRV_CONSISTENT") is expected
