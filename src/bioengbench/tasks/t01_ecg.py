"""t01: denoise an ambulatory ECG, detect R peaks, report heart-rate variability.

Case generation uses the McSharry dynamical model (bioengbench.models.ecg)
with per-case variation in heart rate, rhythm, morphology and amplitude,
then adds three contaminants found in real ambulatory recordings:
baseline wander (respiration and electrode motion), mains interference
near 60 Hz, and broadband electrode noise.

Every tolerance below states its reason. Tolerances come from physics and
clinical practice, never from the reference solution's measured error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from bioengbench.checks.io import FloatArray, read_csv_columns, read_json_numbers
from bioengbench.checks.matching import match_events
from bioengbench.checks.signal import band_power, bandlimited_rmse
from bioengbench.checks.units import Tolerance, quantity
from bioengbench.core.registry import register
from bioengbench.core.seeds import streams
from bioengbench.core.task import TaskPlugin
from bioengbench.models.ecg import STANDARD_WAVES, RhythmParams, Wave, simulate

FS = 360.0  # Hz, as in the MIT-BIH Arrhythmia Database
DURATION_S = 60.0
EDGE_S = 2.0  # ignored at both ends: filter transients are legitimate there

FILTERED = ("time_s", "ecg_mV")
PEAKS = ("sample_index", "time_s")
METRIC_KEYS = ("mean_hr_bpm", "sdnn_ms", "rmssd_ms", "n_beats")

# --- tolerances and their justification -----------------------------------------
BASELINE_BAND = (0.05, 0.40)  # Hz, where the simulated wander lives
LINE_BAND = (59.0, 61.0)  # Hz, mains drift is at most +/-0.05 Hz
BASELINE_MAX_RATIO = 0.10  # >= 10 dB suppression of wander
LINE_MAX_RATIO = 0.05  # >= 13 dB suppression of mains
MORPH_BAND = (1.0, 40.0)  # Hz, diagnostic morphology band (AHA bandwidth guidance)
# In-band noise floor is sigma*sqrt(39/180) <= 0.017 mV; 0.05 mV is about 4 % of a
# 1.2 mV R wave and is exceeded by over-filtering (a 5 Hz high-pass flattens the T wave).
MORPH_MAX_RMSE_MV = 0.05
AMP_REL_TOL = 0.15  # R amplitude must survive filtering; also catches V vs mV
# 40 ms = 14.4 samples. Beat-annotation convention (ANSI/AAMI EC57) uses 150 ms;
# 40 ms is stricter so a peak on the QRS upstroke still counts but a peak on
# the integrator output (about 75 ms late) does not.
PEAK_TOL_S = 0.040
PEAK_MIN = 0.98  # sensitivity and precision
TIME_INDEX_TOL_S = 1e-4  # time_s must equal sample_index / fs
# Peak localisation error of +/-1 sample (2.8 ms) bounds RR error near 4 ms, so
# SDNN and RMSSD get a 3 ms absolute floor plus a relative band.
HR_TOL = Tolerance(quantity(1.0, "beat/minute"))
SDNN_TOL = Tolerance(quantity(3.0, "ms"), 0.10)
RMSSD_TOL = Tolerance(quantity(3.0, "ms"), 0.15)
NBEATS_TOL = 2  # beats in the first or last second may be missed legitimately
CONSISTENCY_TOL_MS = 1.0  # reported HRV must follow from the reported peaks


@dataclass(frozen=True)
class EcgCase:
    fs: float
    t: FloatArray
    raw: FloatArray
    clean: FloatArray
    r_idx: np.ndarray  # int64 sample indices of true R peaks
    params: dict[str, float]


def hrv(peaks: np.ndarray, fs: float) -> dict[str, float]:
    """HRV from R-peak sample indices. RR in ms; SDNN uses the sample SD (n - 1)."""
    rr = np.diff(np.sort(peaks)) / fs * 1000.0
    return {
        "mean_hr_bpm": float(60000.0 / rr.mean()),
        "sdnn_ms": float(rr.std(ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        "n_beats": float(len(peaks)),
    }


def _waves(rng: np.random.Generator) -> dict[str, Wave]:
    """Per-patient morphology: jitter the standard waves; T amplitude varies most."""
    out = {}
    for name, w in STANDARD_WAVES.items():
        theta = w.theta if name == "R" else w.theta + rng.uniform(-0.05, 0.05)
        a = w.a * rng.uniform(0.85, 1.15) * (rng.uniform(0.6, 1.4) if name == "T" else 1.0)
        out[name] = Wave(theta, a, w.b * rng.uniform(0.9, 1.1))
    return out


def _true_peaks(clean: FloatArray, r_times: FloatArray, fs: float) -> np.ndarray:
    """R peak = local maximum of the clean signal within 22 ms of each R phase event."""
    half = 8
    idx = []
    for c in np.round(r_times * fs).astype(int):
        lo, hi = c - half, c + half + 1
        if lo >= 0 and hi <= len(clean):
            idx.append(lo + int(np.argmax(clean[lo:hi])))
    return np.array(idx, dtype=np.int64)


def make_case(seed: int) -> EcgCase:
    rng_rhythm, rng_morph, rng_scale, rng_noise, rng_sim = streams(seed, 5)
    rhythm = RhythmParams(
        mean_hr_bpm=rng_rhythm.uniform(55, 100),
        resp_hz=rng_rhythm.uniform(0.15, 0.40),
        rsa_frac=rng_rhythm.uniform(0.02, 0.06),
        mayer_frac=rng_rhythm.uniform(0.01, 0.04),
        ar_sd_frac=rng_rhythm.uniform(0.005, 0.02),
        phase_rsa=rng_rhythm.uniform(0, 2 * np.pi),
        phase_mayer=rng_rhythm.uniform(0, 2 * np.pi),
    )
    sim = simulate(rhythm, _waves(rng_morph), DURATION_S, FS, rng_sim)
    t = sim.t

    # Respiration modulates QRS amplitude (thoracic impedance changes).
    am_depth = rng_scale.uniform(0.05, 0.15)
    am_phase = rng_scale.uniform(0, 2 * np.pi)
    z = sim.z * (1.0 + am_depth * np.sin(2 * np.pi * rhythm.resp_hz * t + am_phase))
    r_peaks = _true_peaks(z, sim.r_times, FS)
    r_amp = rng_scale.uniform(0.8, 1.6)  # mV
    clean = z * (r_amp / float(np.median(z[r_peaks])))

    wander_amp = rng_noise.uniform(0.4, 1.0)
    f_w = rng_noise.uniform(0.05, 0.35, size=3)
    ph_w = rng_noise.uniform(0, 2 * np.pi, size=3)
    wander = sum(
        w * wander_amp * np.sin(2 * np.pi * f * t + p)
        for w, f, p in zip((1.0, 0.6, 0.4), f_w, ph_w, strict=True)
    )
    line_hz = 60.0 + rng_noise.uniform(-0.05, 0.05)
    line_amp = rng_noise.uniform(0.10, 0.30)
    line = line_amp * np.sin(2 * np.pi * line_hz * t + rng_noise.uniform(0, 2 * np.pi))
    noise_sd = rng_noise.uniform(0.015, 0.035)
    raw = clean + wander + line + rng_noise.normal(0.0, noise_sd, size=len(t))

    params = {
        "hr_bpm": rhythm.mean_hr_bpm,
        "r_amp_mV": r_amp,
        "wander_amp_mV": wander_amp,
        "line_hz": line_hz,
        "line_amp_mV": line_amp,
        "noise_sd_mV": noise_sd,
    }
    return EcgCase(FS, t, raw, clean, r_peaks, params)


def write_inputs(case: EcgCase, input_dir: Path) -> None:
    np.savetxt(
        input_dir / "ecg_raw.csv",
        np.column_stack([case.t, case.raw]),
        delimiter=",",
        header="time_s,ecg_mV",
        comments="",
        fmt="%.6f",
    )
    meta = {"fs_hz": case.fs, "duration_s": DURATION_S, "units": {"time": "s", "voltage": "mV"}}
    (input_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> EcgCase:
    case: EcgCase = ctx.truth
    return case


def _interior(case: EcgCase) -> tuple[int, int]:
    e = int(EDGE_S * case.fs)
    return e, len(case.raw) - e


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    names = ("ecg_filtered.csv", "rpeaks.csv", "metrics.json")
    missing = [n for n in names if not (ctx.output_dir / n).is_file()]
    return Outcome(not missing, f"missing: {', '.join(missing)}" if missing else "")


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    ctx.outputs["filtered"] = read_csv_columns(ctx.output_dir / "ecg_filtered.csv", FILTERED)
    ctx.outputs["peaks"] = read_csv_columns(ctx.output_dir / "rpeaks.csv", PEAKS)
    ctx.outputs["metrics"] = read_json_numbers(ctx.output_dir / "metrics.json", METRIC_KEYS)
    return Outcome(True)


@check("TIME_AXIS", gate=True)
def time_axis(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    t = ctx.outputs["filtered"]["time_s"]
    if len(t) != len(case.t):
        return Outcome(False, f"{len(t)} samples, expected {len(case.t)}")
    err = float(np.max(np.abs(t - case.t)))
    return Outcome(err <= 1e-5, "time_s does not match the input time axis", err, 1e-5)


@check("BASELINE")
def baseline(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    y = ctx.outputs["filtered"]["ecg_mV"]
    ratio = band_power(y, case.fs, BASELINE_BAND) / band_power(case.raw, case.fs, BASELINE_BAND)
    return Outcome(
        ratio <= BASELINE_MAX_RATIO, "baseline wander not removed", ratio, BASELINE_MAX_RATIO
    )


@check("POWERLINE")
def powerline(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    y = ctx.outputs["filtered"]["ecg_mV"]
    ratio = band_power(y, case.fs, LINE_BAND) / band_power(case.raw, case.fs, LINE_BAND)
    return Outcome(ratio <= LINE_MAX_RATIO, "60 Hz interference not removed", ratio, LINE_MAX_RATIO)


@check("FIDELITY")
def fidelity(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    y = ctx.outputs["filtered"]["ecg_mV"]
    rmse = bandlimited_rmse(y, case.clean, case.fs, MORPH_BAND, int(EDGE_S * case.fs))
    return Outcome(
        rmse <= MORPH_MAX_RMSE_MV, "1-40 Hz morphology distorted", rmse, MORPH_MAX_RMSE_MV
    )


@check("AMPLITUDE")
def amplitude(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    y = ctx.outputs["filtered"]["ecg_mV"]
    lo, hi = _interior(case)
    r = case.r_idx[(case.r_idx > lo) & (case.r_idx < hi)]
    true_amp = float(np.median(case.clean[r] - np.mean(case.clean[lo:hi])))
    out_amp = float(np.median(y[r] - np.mean(y[lo:hi])))
    rel = abs(out_amp - true_amp) / true_amp
    return Outcome(
        rel <= AMP_REL_TOL, f"R amplitude {out_amp:.3f} mV vs {true_amp:.3f} mV", rel, AMP_REL_TOL
    )


def _detected(ctx: CheckContext) -> np.ndarray:
    idx = ctx.outputs["peaks"]["sample_index"]
    return np.asarray(np.round(idx), dtype=np.int64)


@check("PEAK_FORMAT", gate=True)
def peak_format(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    raw = ctx.outputs["peaks"]["sample_index"]
    idx = _detected(ctx)
    if not np.allclose(raw, idx):
        return Outcome(False, "sample_index must be integers")
    if np.any(idx < 0) or np.any(idx >= len(case.raw)):
        return Outcome(False, "sample_index outside the recording")
    if np.any(np.diff(idx) <= 0):
        return Outcome(False, "sample_index must be strictly increasing")
    err = float(np.max(np.abs(ctx.outputs["peaks"]["time_s"] - idx / case.fs)))
    return Outcome(
        err <= TIME_INDEX_TOL_S, "time_s != sample_index / fs (0-based?)", err, TIME_INDEX_TOL_S
    )


def _peak_counts(ctx: CheckContext):  # type: ignore[no-untyped-def]
    case = _case(ctx)
    lo, hi = _interior(case)
    det = _detected(ctx)
    true_i = case.r_idx[(case.r_idx > lo) & (case.r_idx < hi)]
    det_i = det[(det > lo) & (det < hi)]
    return match_events(true_i, det_i, round(PEAK_TOL_S * case.fs))


@check("PEAK_SENS")
def peak_sensitivity(ctx: CheckContext) -> Outcome:
    s = _peak_counts(ctx).sensitivity
    return Outcome(s >= PEAK_MIN, "R peaks missed", s, PEAK_MIN)


@check("PEAK_PREC")
def peak_precision(ctx: CheckContext) -> Outcome:
    p = _peak_counts(ctx).precision
    return Outcome(p >= PEAK_MIN, "false R-peak detections", p, PEAK_MIN)


def _hrv_check(code: str, key: str, unit: str, tol: Tolerance):  # type: ignore[no-untyped-def]
    @check(code)
    def run(ctx: CheckContext) -> Outcome:
        case = _case(ctx)
        truth = quantity(hrv(case.r_idx, case.fs)[key], unit)
        value = quantity(ctx.outputs["metrics"][key], unit)
        ok = tol.within(value, truth)
        err = float(abs((value - truth).to(unit).magnitude))
        return Outcome(
            ok,
            f"{key} {value.magnitude:.3f} vs true {truth.magnitude:.3f}",
            err,
            tol.allowed(truth).to(unit).magnitude,
        )

    return run


@check("NBEATS")
def n_beats(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    err = abs(ctx.outputs["metrics"]["n_beats"] - len(case.r_idx))
    return Outcome(err <= NBEATS_TOL, "beat count wrong", err, NBEATS_TOL)


@check("HRV_CONSISTENT")
def hrv_consistent(ctx: CheckContext) -> Outcome:
    """Reported metrics must follow from the reported peaks (catches hard-coded or
    mis-defined metrics independently of peak accuracy, for failure diagnosis)."""
    case = _case(ctx)
    own = hrv(_detected(ctx), case.fs)
    met = ctx.outputs["metrics"]
    worst = max(abs(met[k] - own[k]) for k in ("sdnn_ms", "rmssd_ms"))
    return Outcome(
        worst <= CONSISTENCY_TOL_MS,
        "metrics do not match the reported peaks",
        worst,
        CONSISTENCY_TOL_MS,
    )


@register
class EcgTask(TaskPlugin):
    task_id = "t01-ecg-denoise-rpeaks"
    version = "2.0.0"  # 2.x: McSharry generator, 128-bit seeds, O(n+m) matcher

    def make_case(self, seed: int) -> EcgCase:
        return make_case(seed)

    def write_inputs(self, case: EcgCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def languages(self) -> tuple[str, ...]:
        return ("python", "octave")

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        if lang == "octave":
            return ("{submission}", "{input_dir}/ecg_raw.csv", "360", "{output_dir}")
        return (
            "{submission}", "--input", "{input_dir}/ecg_raw.csv",
            "--fs", "360", "--outdir", "{output_dir}",
        )  # fmt: skip

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t01",
            [
                CheckGroup("load", [files_present, schema, time_axis], gate=True),
                CheckGroup("signal", [baseline, powerline, fidelity, amplitude]),
                CheckGroup("peaks", [peak_format, peak_sensitivity, peak_precision]),
                CheckGroup(
                    "hrv",
                    [
                        _hrv_check("HR", "mean_hr_bpm", "beat/minute", HR_TOL),
                        _hrv_check("SDNN", "sdnn_ms", "ms", SDNN_TOL),
                        _hrv_check("RMSSD", "rmssd_ms", "ms", RMSSD_TOL),
                        n_beats,
                        hrv_consistent,
                    ],
                ),
            ],
        )
