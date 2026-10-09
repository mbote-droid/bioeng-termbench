"""False-accept side: each realistic mistake must trigger its matching check code.

Every case starts from a perfect submission built from ground truth and
changes exactly one thing.
"""

import numpy as np
import pytest
from conftest import write_t01_outputs
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

from bioengbench.core.scoring import Status
from bioengbench.tasks.t01_ecg import hrv

SEED = 0x6A3F_19C2_7B44_0D85_A1E0_3C77_59F2_8B16


@pytest.fixture(scope="module")
def case(t01):
    return t01.make_case(SEED)


def _codes(t01, out, case) -> set[str]:
    return {r.code for r in t01.evaluate_outputs(out, case) if r.status is Status.FAIL}


def test_perfect_submission_passes(t01, case, tmp_path) -> None:
    results = t01.evaluate_outputs(write_t01_outputs(tmp_path, case), case)
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_unfiltered_signal(t01, case, tmp_path) -> None:
    out = write_t01_outputs(tmp_path, case, filtered=case.raw)
    assert {"BASELINE", "POWERLINE"} <= _codes(t01, out, case)


def test_notch_only_leaves_wander(t01, case, tmp_path) -> None:
    b, a = iirnotch(60.0, 30.0, fs=case.fs)
    out = write_t01_outputs(tmp_path, case, filtered=sosfiltfilt(tf2sos(b, a), case.raw))
    assert "BASELINE" in _codes(t01, out, case)


def test_volts_instead_of_millivolts(t01, case, tmp_path) -> None:
    out = write_t01_outputs(tmp_path, case, filtered=case.clean / 1000.0)
    assert "AMPLITUDE" in _codes(t01, out, case)


def test_overfiltering_5hz_highpass_distorts_morphology(t01, case, tmp_path) -> None:
    sos = butter(2, 5.0, btype="highpass", fs=case.fs, output="sos")
    out = write_t01_outputs(tmp_path, case, filtered=sosfiltfilt(sos, case.clean))
    assert "FIDELITY" in _codes(t01, out, case)


def test_peaks_from_delayed_detection_signal(t01, case, tmp_path) -> None:
    late = case.r_idx + int(0.075 * case.fs)  # integrator output peaks about 75 ms late
    out = write_t01_outputs(tmp_path, case, peaks=late[late < len(case.t)])
    assert {"PEAK_SENS", "PEAK_PREC"} <= _codes(t01, out, case)


def test_every_other_beat_missed(t01, case, tmp_path) -> None:
    out = write_t01_outputs(tmp_path, case, peaks=case.r_idx[::2])
    assert "PEAK_SENS" in _codes(t01, out, case)


def test_t_waves_detected_as_beats(t01, case, tmp_path) -> None:
    t_waves = case.r_idx + int(0.25 * case.fs)
    both = np.sort(np.concatenate([case.r_idx, t_waves[t_waves < len(case.t)]]))
    out = write_t01_outputs(tmp_path, case, peaks=both)
    assert "PEAK_PREC" in _codes(t01, out, case)


def test_one_based_indices(t01, case, tmp_path) -> None:
    out = write_t01_outputs(tmp_path, case)
    peaks = np.column_stack([case.r_idx + 1, case.r_idx / case.fs])  # index shifted, time not
    np.savetxt(
        out / "rpeaks.csv",
        peaks,
        delimiter=",",
        header="sample_index,time_s",
        comments="",
        fmt=["%d", "%.6f"],
    )
    assert "PEAK_FORMAT" in _codes(t01, out, case)


@pytest.mark.parametrize(
    ("key", "factor", "code"),
    [("sdnn_ms", 1e-3, "SDNN"), ("rmssd_ms", 1e-3, "RMSSD"), ("mean_hr_bpm", 1 / 60, "HR")],
)
def test_metric_unit_slips(t01, case, tmp_path, key, factor, code) -> None:
    m = hrv(case.r_idx, case.fs)
    m[key] *= factor
    assert code in _codes(t01, write_t01_outputs(tmp_path, case, metrics=m), case)


def test_population_sd_instead_of_sample_sd_is_within_tolerance(t01, case, tmp_path) -> None:
    """ddof=0 vs ddof=1 changes SDNN by a factor sqrt((n-1)/n), about 0.7 % here.
    Below the 10 % band, so the grader does not penalise it: a documented, deliberate
    insensitivity (the instruction still specifies n - 1)."""
    m = hrv(case.r_idx, case.fs)
    m["sdnn_ms"] *= np.sqrt((len(case.r_idx) - 2) / (len(case.r_idx) - 1))
    assert "SDNN" not in _codes(t01, write_t01_outputs(tmp_path, case, metrics=m), case)


def test_hard_coded_metrics_are_inconsistent_with_peaks(t01, case, tmp_path) -> None:
    m = {"mean_hr_bpm": 70.0, "sdnn_ms": 50.0, "rmssd_ms": 40.0, "n_beats": 70}
    codes = _codes(t01, write_t01_outputs(tmp_path, case, metrics=m), case)
    assert "HRV_CONSISTENT" in codes


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda out: (out / "metrics.json").unlink(), "FILES"),
        (lambda out: (out / "rpeaks.csv").write_text("idx,t\n1,2\n"), "SCHEMA"),
        (lambda out: (out / "metrics.json").write_text('{"mean_hr_bpm": "70"}'), "SCHEMA"),
        (lambda out: (out / "metrics.json").write_text('{"mean_hr_bpm": NaN}'), "SCHEMA"),
    ],
)
def test_malformed_outputs_fail_closed(t01, case, tmp_path, mutate, code) -> None:
    out = write_t01_outputs(tmp_path, case)
    mutate(out)
    results = t01.evaluate_outputs(out, case)
    assert code in {r.code for r in results if r.status is Status.FAIL}
    assert not any(r.passed for r in results if r.code not in {"FILES", "SCHEMA"})


def test_truncated_trace_fails_time_axis(t01, case, tmp_path) -> None:
    out = write_t01_outputs(tmp_path, case)
    lines = (out / "ecg_filtered.csv").read_text().splitlines()
    (out / "ecg_filtered.csv").write_text("\n".join(lines[:-10]) + "\n")
    assert "TIME_AXIS" in _codes(t01, out, case)
