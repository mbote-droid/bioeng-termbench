"""Every submission the validation campaign grades, in one registry.

A probe is a submission with an expectation:

  reference  a correct solution (often a different method or language):
             must pass every case. Failures estimate the false-reject rate.
  negative   a realistic mistake: must fail, and with the expected codes.
             Passing cases estimate the false-accept rate.
  boundary   a defensible-looking method the task contract excludes: must
             fail on exactly the expected codes.
  redteam    an attack on the grader itself (see redteam/): must fail RUN
             or be harmless, and must not affect anything outside its case.

Negative probes are mostly *mutants*: the reference with one idea changed by
a text edit, so each differs from a passing program in exactly one way. The
edit must apply (checked), otherwise a refactor of the reference would
silently turn a negative probe into a second reference.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
REDTEAM = Path(__file__).resolve().parent / "redteam"

Edit = tuple[str, str]


@dataclass(frozen=True)
class Probe:
    task: str
    name: str
    kind: str  # reference | negative | boundary | redteam
    note: str
    source: Path  # file or package directory
    edits: tuple[Edit, ...] = ()
    lang: str = "python"
    expected: frozenset[str] = field(default_factory=frozenset)
    seeds: int | None = None  # override the campaign default
    replace_files: tuple[tuple[str, Path], ...] = ()  # (name in package, source file)

    @property
    def key(self) -> str:
        return f"{self.task}/{self.name}"

    def build(self, workdir: Path) -> Path:
        """Materialise the submission under `workdir`; return its path."""
        dest = workdir / "probe" / self.source.name
        if dest.exists():
            shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.source.is_dir():
            shutil.copytree(self.source, dest, ignore=shutil.ignore_patterns("__pycache__"))
            for name, src in self.replace_files:
                shutil.copyfile(src, dest / name)
            target = dest / self.edits_file if self.edits else None
        else:
            shutil.copyfile(self.source, dest)
            target = dest
        if self.edits:
            assert target is not None
            text = target.read_text()
            for old, new in self.edits:
                if old not in text:
                    raise ValueError(f"{self.key}: edit does not apply: {old[:60]!r}")
                text = text.replace(old, new)
            target.write_text(text)
        return dest

    @property
    def edits_file(self) -> str:
        return "solver.py"  # only package probes with edits use this (t05)


def _ref(
    task: str, name: str, path: str, note: str, lang: str = "python", seeds: int | None = None
) -> Probe:
    return Probe(task, name, "reference", note, TASKS / task / path, lang=lang, seeds=seeds)


def _mut(
    task: str,
    name: str,
    base: str,
    edits: list[Edit],
    expected: set[str],
    note: str,
    kind: str = "negative",
    seeds: int | None = None,
) -> Probe:
    return Probe(
        task,
        name,
        kind,
        note,
        TASKS / task / base,
        tuple(edits),
        expected=frozenset(expected),
        seeds=seeds,
    )


T01, T02, T03, T04, T05 = (
    "t01-ecg-denoise-rpeaks",
    "t02-pk-fit-dose",
    "t03-glucose-closed-loop",
    "t04-bioheat-ablation",
    "t05-pk-debug",
)
T01_SOL, T02_SOL, T04_SOL = (
    "solution/pipeline.py",
    "solution/pk_pipeline.py",
    "solution/ablation.py",
)
T03_CONTROLS = ROOT / "tests/tasks/t03_controllers"

PROBES: list[Probe] = [
    # --- t01 ECG ------------------------------------------------------------
    _ref(T01, "reference", T01_SOL, "Butterworth + notch, Pan-Tompkins"),
    _ref(
        T01,
        "spline_hilbert",
        "references/pipeline_spline_hilbert.py",
        "least-squares mains, spline baseline, Hilbert detector",
    ),
    _ref(
        T01, "octave_rbj", "references/pipeline_rbj.m", "hand-built biquads, Octave", "octave", 20
    ),
    Probe(
        T01,
        "median_baseline",
        "boundary",
        "median-filter baseline: T-wave distortion near the 0.05 mV limit",
        TASKS / T01 / "references/pipeline_median_hilbert.py",
        expected=frozenset({"FIDELITY"}),
    ),
    _mut(
        T01,
        "volts",
        T01_SOL,
        [("y = denoise(x, a.fs)", "y = denoise(x, a.fs) / 1000.0")],
        {"AMPLITUDE"},
        "filtered ECG written in volts",
    ),
    _mut(
        T01,
        "no_notch",
        T01_SOL,
        [("return sosfiltfilt(tf2sos(b, a), y)", "return y")],
        {"POWERLINE"},
        "mains interference left in",
    ),
    _mut(
        T01,
        "no_highpass",
        T01_SOL,
        [("y = sosfiltfilt(sos_hp, x)", "y = x")],
        {"BASELINE"},
        "baseline wander left in",
    ),
    _mut(
        T01,
        "unrefined_peaks",
        T01_SOL,
        [("peaks.append(lo + int(np.argmax(y[lo:hi])))", "peaks.append(c)")],
        set(),
        "peaks at the integrator maximum, unrefined: within 40 ms, so acceptable",
        kind="reference",  # equivalent mutant: designed as a mistake, passed 100/100
    ),
    _mut(
        T01,
        "hr_per_second",
        T01_SOL,
        [('"mean_hr_bpm": float(60000.0 / rr.mean())', '"mean_hr_bpm": float(1000.0 / rr.mean())')],
        {"HR"},
        "heart rate in beats per second",
    ),
    _mut(
        T01,
        "sdnn_in_seconds",
        T01_SOL,
        [('"sdnn_ms": float(rr.std(ddof=1))', '"sdnn_ms": float(rr.std(ddof=1) / 1000.0)')],
        {"HRV_CONSISTENT"},
        "SDNN in s labelled ms",
    ),
    # --- t02 pharmacokinetics -------------------------------------------------
    _ref(T02, "reference", T02_SOL, "exact expm model, multi-start TRF"),
    _ref(T02, "octave", "references/pk_pipeline.m", "Octave, Nelder-Mead", "octave", 6),
    _mut(
        T02,
        "linear_se",
        T02_SOL,
        [("se = np.sqrt(np.abs(np.diag(", "se = p * np.sqrt(np.abs(np.diag(")],
        {"SE_CONSISTENT"},
        "standard errors of parameters, not of their logs",
        seeds=6,
    ),
    _mut(
        T02,
        "no_ln2",
        T02_SOL,
        [("float(np.log(2) / eig[0])", "float(1.0 / eig[0])")],
        {"HALF_LIVES"},
        "time constant reported as half-life",
        seeds=6,
    ),
    _mut(
        T02,
        "daily_as_dose",
        T02_SOL,
        [('"dose_mg": dose,', '"dose_mg": dose * 24 / tau,')],
        {"REGIMEN_SELF"},
        "daily dose reported per infusion",
        seeds=6,
    ),
    _mut(
        T02,
        "no_safety_margin",
        T02_SOL,
        [
            (
                'lo, hi = prob["design_targets_mg_per_L"]',
                'lo, hi = prob["therapeutic_window_mg_per_L"]',
            )
        ],
        {"REGIMEN_TARGETS"},
        "designs to the window edges, ignoring estimation error",
        seeds=40,
    ),
    _mut(
        T02,
        "single_start",
        T02_SOL,
        [("for _ in range(12)", "for _ in range(0)")],
        set(),
        "one optimiser start: converged on every case",
        kind="reference",  # equivalent mutant: designed as a mistake, passed 12/12
        seeds=12,
    ),
    # --- t03 closed-loop glucose ----------------------------------------------
    _ref(T03, "reference", "solution/controller.py", "Kalman + PID + IOB + suspend + exercise"),
    _ref(T03, "octave", "references/controller.m", "Octave port", "octave", 10),
    *[
        Probe(T03, name, "negative", note, T03_CONTROLS / f"{name}.py", expected=frozenset(exp),
              seeds=20 if exp == {"RUN"} else None)  # protocol failures are deterministic
        for name, exp, note in [
            ("basal_only", {"TIME_IN_RANGE", "CORRECTION_TIME"}, "open loop"),
            ("aggressive_p", {"HYPO"}, "5x gain, no safety layers"),
            ("integral_only", {"HYPO", "SETTLING"}, "phase lag: limit cycle"),
            ("naive_pid", {"HYPO"}, "plausible PID without safety layers"),
            ("over_max", {"ACTUATOR"}, "requests beyond the pump"),
            ("crash_on_dropout", {"RUN"}, "None arithmetic on dropout"),
            ("garbage", {"RUN"}, "non-JSON reply"),
            ("string_rate", {"RUN"}, "rate as a string"),
            ("slow", {"RUN"}, "misses the step deadline"),
            ("silent", {"RUN"}, "never replies"),
        ]
    ],
    # --- t04 bioheat ----------------------------------------------------------
    _ref(T04, "reference", T04_SOL, "cell-centred FV, Crank-Nicolson + Picard"),
    _ref(T04, "fem", "references/ablation_fem.py", "vertex-centred P1 FEM, lumped", seeds=8),
    _ref(T04, "octave", "references/ablation.m", "Octave, sparse solves, fzero", "octave", 4),
    _mut(
        T04,
        "lagged_perfusion",
        T04_SOL,
        [("for _ in range(3):", "for _ in range(1):")],
        {"CONVERGENCE_TIME"},
        "perfusion lagged one step: first order in time",
        seeds=4,
    ),
    _mut(
        T04,
        "cartesian",
        T04_SOL,
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
        "slab geometry instead of sphere",
        seeds=4,
    ),
    _mut(
        T04,
        "no_collapse",
        T04_SOL,
        [("return expit((stasis_c - t_body - th) / stasis_w)", "return np.ones_like(th)")],
        {"ACCURACY"},
        "perfusion independent of temperature",
        seeds=4,
    ),
    _mut(
        T04,
        "ignores_resolution",
        T04_SOL,
        [
            (
                "res = simulate(cfg, a.power, a.heating_time, a.perfusion, a.nr, a.dt)",
                "res = simulate(cfg, a.power, a.heating_time, a.perfusion, 400, 1.0)",
            )
        ],
        {"CONVERGENCE_SPACE", "CONVERGENCE_TIME"},
        "ignores --nr/--dt",
        seeds=4,
    ),
    _mut(
        T04,
        "budget_wrong_time",
        T04_SOL,
        [('"stored": cap @ th', '"stored": e_on')],
        {"ENERGY_BUDGET"},
        "stored energy at the wrong time",
        seeds=4,
    ),
    _mut(
        T04,
        "nominal_only",
        T04_SOL,
        [
            (
                'lo, hi = cfg["tissue"]["perfusion_range_per_s"]',
                'lo, hi = [sum(cfg["tissue"]["perfusion_range_per_s"]) / 2] * 2',
            )
        ],
        {"COVERAGE"},
        "plans for nominal perfusion only",
        seeds=4,
    ),
    _mut(
        T04,
        "longest_time",
        T04_SOL,
        [("p * t < best[0] * best[1]", "t > best[1]")],
        {"EFFICIENCY"},
        "always the longest heating",
        seeds=4,
    ),
    _mut(
        T04,
        "inconsistent_prediction",
        T04_SOL,
        [
            (
                "at_hi = simulate(cfg, p, t, hi, 400, 1.0)",
                "at_hi = simulate(cfg, p, t, 0.5 * (lo + hi), 400, 1.0)",
            )
        ],
        {"PLAN_CONSISTENT"},
        "predicts at the nominal perfusion",
        seeds=12,
    ),
    # --- t05 debugging --------------------------------------------------------
    Probe(T05, "fixed", "reference", "all five defects fixed", TASKS / T05 / "solution/pkmodel"),
    Probe(
        T05,
        "shipped",
        "negative",
        "the package as given to the agent",
        TASKS / T05 / "environment/pkmodel",
        expected=frozenset(
            {"STANDARD", "SHORT_INFUSION", "BOLUS", "LONG_HORIZON", "COINCIDENT_DOSES"}
        ),
    ),
    *[
        Probe(
            T05,
            f"only_{name}",
            "negative",
            note,
            TASKS / T05 / "solution/pkmodel",
            expected=frozenset(exp),
            replace_files=((module, TASKS / T05 / "environment/pkmodel" / module),),
        )
        for name, module, exp, note in [
            (
                "units",
                "units.py",
                {"STANDARD", "SHORT_INFUSION", "BOLUS", "LONG_HORIZON", "COINCIDENT_DOSES"},
                "CL converted the wrong way",
            ),
            ("float32_grid", "grid.py", {"LONG_HORIZON"}, "output grid in float32"),
            ("doses_by_time", "regimen.py", {"COINCIDENT_DOSES"}, "doses keyed by start time"),
            (
                "adaptive_solver",
                "solver.py",
                {"STANDARD", "SHORT_INFUSION", "BOLUS"},
                "loose RK45 over the horizon, no boluses",
            ),
        ]
    ],
    Probe(
        T05,
        "only_bolus_skipped",
        "negative",
        "boluses ignored",
        TASKS / T05 / "solution/pkmodel",
        (
            (
                "        return sum(d.amount_mg for d in doses if d.is_bolus and d.start_h == t)",
                "        return 0.0",
            ),
        ),
        expected=frozenset({"BOLUS"}),
    ),
]


def by_key() -> dict[str, Probe]:
    keys = [p.key for p in PROBES]
    assert len(keys) == len(set(keys)), "duplicate probe names"
    return {p.key: p for p in PROBES}


Builder = Callable[[Path], Path]
