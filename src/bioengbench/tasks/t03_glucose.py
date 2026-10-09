"""t03: closed-loop insulin delivery (an "artificial pancreas") for type 1 diabetes.

The agent writes a controller. The grader keeps the virtual patient (Bergman
minimal model, hidden parameters) in its own process and runs a
software-in-the-loop test: every 5 minutes it sends the controller a CGM
reading over a line protocol and applies the returned insulin rate to the
patient. The controller never sees the patient's state or parameters, only
what a real pump sees: the CGM reading (sometimes missing or corrupted), the
time, the programmed basal rate and the pump's maximum rate.

Two scenarios per hidden patient, each with a fresh controller process:

  A. 24 h with three unannounced meals (30-60 g), one 30 min sensor dropout,
     one false-high sensor spike and one compression low (false low) overnight.
  B. 6 h hyperglycaemia correction from 250 mg/dL, fasting.

Faults come from the FMEA in docs/FMEA.md; each high-risk failure mode has a
scenario here. Thresholds follow the international consensus on CGM targets
(Battelino et al., Diabetes Care 2019): time in range 70-180 mg/dL above 70 %,
time above 250 mg/dL below 5 %, and no level-1 hypoglycaemia (< 70 mg/dL)
anywhere, which is stricter than the consensus (< 4 %) because a controller
under test should not cause any.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bioengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from bioengbench.checks.io import FloatArray, read_csv_columns
from bioengbench.core.registry import register
from bioengbench.core.runner import ProtocolError, Runner
from bioengbench.core.seeds import streams
from bioengbench.core.task import Execution, RunPlan, TaskPlugin
from bioengbench.models.glucose import (
    SAMPLE_MIN,
    Cgm,
    Disturbances,
    Exercise,
    Meal,
    Patient,
    SensorFault,
    advance,
    initial_state,
)

U_MAX = 120.0  # mU/min (7.2 U/h), pump limit
FIRST_STEP_TIMEOUT_S = 30.0  # interpreter start-up
STEP_TIMEOUT_S = 2.0  # per 5 min sample, generous for any controller
DAY_MIN = 1440.0
CORRECTION_MIN = 360.0
CORRECTION_START = 250.0

HYPO_LIMIT = 70.0  # mg/dL, level-1 hypoglycaemia
TIR_RANGE = (70.0, 180.0)
TIR_MIN = 0.70
TAR250_MAX = 0.05
CORRECTION_LIMIT_MIN = 120.0  # must be below 180 mg/dL within 2 h
SETTLE_WINDOW_MIN = 120.0  # last 2 h of the correction scenario
SETTLE_P2P_MAX = 50.0  # mg/dL; a sustained oscillation (limit cycle) exceeds this
SETTLE_MEAN = (80.0, 160.0)

EXERCISE_MIN = 45.0
ANNOUNCE_LEAD_MIN = 60.0
EXERCISE_UPTAKE = (0.004, 0.008)  # 1/min, insulin-independent uptake during the bout
EXERCISE_GAIN = (0.5, 1.0)  # insulin action amplification at the end of the bout

SCENARIOS = ("day", "correction")
TRACE = ("t_min", "plasma_glucose_mg_dL")
PUMP = (
    "t_min",
    "cgm_valid",
    "cgm_mg_dL",
    "requested_mU_per_min",
    "delivered_mU_per_min",
)  # cgm_mg_dL is 0 where cgm_valid is 0 (dropout)


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_min: float
    g0: float
    disturbances: Disturbances
    faults: tuple[SensorFault, ...]

    def exercise_announced(self, t: float) -> bool:
        """The user announces exercise ANNOUNCE_LEAD_MIN before it starts (as
        commercial systems' activity modes expect); the flag stays on until the
        bout ends. Post-exercise sensitivity is not announced."""
        return any(
            e.start_min - ANNOUNCE_LEAD_MIN <= t < e.start_min + e.duration_min
            for e in self.disturbances.exercise
        )


@dataclass(frozen=True)
class GlucoseCase:
    seed: int
    patient: Patient
    scenarios: tuple[Scenario, ...]


def sample_patient(rng: np.random.Generator) -> Patient:
    """Virtual patient. The basal rate scales inversely with insulin sensitivity
    (insulin-resistant patients need more insulin), so loop gain S_I * ub stays
    within a clinically plausible range, as basal rates set by clinicians do."""
    mass = rng.uniform(55.0, 95.0)
    p2 = rng.uniform(0.02, 0.03)
    si = rng.uniform(3e-4, 8e-4)
    return Patient(
        body_mass_kg=mass,
        p1=rng.uniform(0.002, 0.008),
        p2=p2,
        p3=si * p2,
        n=rng.uniform(0.09, 0.15),
        gb=rng.uniform(105.0, 135.0),
        ub=0.0085 / si * (mass / 70.0) * rng.uniform(0.8, 1.25),
        tau_s=rng.uniform(5.0, 12.0),
    )


def make_case(seed: int) -> GlucoseCase:
    rng_patient, rng_meals, rng_faults, rng_exercise = streams(seed, 4)
    patient = sample_patient(rng_patient)
    meals = tuple(
        Meal(t0 + rng_meals.uniform(-30, 30), rng_meals.uniform(30, 60), rng_meals.uniform(40, 60))
        for t0 in (60.0, 390.0, 780.0)  # 07:00, 12:30, 19:00 for a 06:00 start
    )
    drop = rng_faults.uniform(120.0, 1000.0)
    spike = rng_faults.uniform(1080.0, 1300.0)
    compression = rng_faults.uniform(1080.0, 1300.0)
    faults = (
        SensorFault(drop, drop + 30.0, "dropout"),
        SensorFault(spike, spike + 15.0, "offset", rng_faults.uniform(60.0, 90.0)),
        SensorFault(compression, compression + 20.0, "offset", -rng_faults.uniform(40.0, 60.0)),
    )
    # Afternoon exercise, start on the 5 min grid so the bout begins at a pump step.
    ex_start = SAMPLE_MIN * round(rng_exercise.uniform(570.0, 630.0) / SAMPLE_MIN)
    exercise = Exercise(
        ex_start,
        EXERCISE_MIN,
        k_uptake=rng_exercise.uniform(*EXERCISE_UPTAKE),
        gain=rng_exercise.uniform(*EXERCISE_GAIN),
    )
    day = Scenario("day", DAY_MIN, patient.gb, Disturbances(meals, (exercise,)), faults)
    correction = Scenario("correction", CORRECTION_MIN, CORRECTION_START, Disturbances(), ())
    return GlucoseCase(seed, patient, (day, correction))


def write_inputs(case: GlucoseCase, input_dir: Path) -> None:
    problem = {
        "sample_period_min": SAMPLE_MIN,
        "basal_rate_mU_per_min": case.patient.ub,
        "pump_max_mU_per_min": U_MAX,
        "units": {"glucose": "mg/dL", "insulin_rate": "mU/min", "time": "min"},
    }
    (input_dir / "problem.json").write_text(json.dumps(problem, indent=2) + "\n")


def _valid_rate(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def run_scenario(
    case: GlucoseCase, scenario: Scenario, session_request: Any, sensor_seed: int
) -> tuple[FloatArray, FloatArray]:
    """Closed loop for one scenario; returns the true trace and the pump log."""
    patient = case.patient
    (sensor_rng,) = streams(sensor_seed, 1)
    cgm = Cgm(sensor_rng, scenario.faults)
    x = initial_state(patient, scenario.g0)
    trace = [(0.0, x[0])]
    pump = []
    t = 0.0
    first = True
    while t < scenario.duration_min - 1e-9:
        reading = cgm.read(float(x[3]), t)
        message = {
            "t_min": t,
            "cgm_mg_dL": None if reading is None else round(reading, 1),
            "basal_mU_per_min": patient.ub,
            "pump_max_mU_per_min": U_MAX,
            "exercise_announced": scenario.exercise_announced(t),
        }
        reply = session_request(message, first)
        first = False
        requested = _valid_rate(reply.get("insulin_mU_per_min"))
        if requested is None:
            raise ProtocolError(f"t={t:g}: 'insulin_mU_per_min' must be a finite number")
        delivered = min(max(requested, 0.0), U_MAX)  # the pump saturates
        x, glucose = advance(patient, x, t, delivered, scenario.disturbances)
        sub_t = t + (np.arange(1, glucose.size + 1)) * (SAMPLE_MIN / glucose.size)
        trace += list(zip(sub_t.tolist(), glucose.tolist(), strict=True))
        pump.append((t, float(reading is not None), reading or 0.0, requested, delivered))
        t += SAMPLE_MIN
    return np.array(trace), np.array(pump)


def write_sim(sim_dir: Path, name: str, trace: FloatArray, pump: FloatArray) -> None:
    for suffix, data, cols in (("trace", trace, TRACE), ("pump", pump, PUMP)):
        np.savetxt(
            sim_dir / f"{name}_{suffix}.csv",
            data,
            delimiter=",",
            header=",".join(cols),
            comments="",
            fmt="%.6f",
        )


# --- checks: they read the grader-written simulation files ----------------------


def _trace(ctx: CheckContext, name: str) -> dict[str, FloatArray]:
    key = f"trace_{name}"
    if key not in ctx.outputs:
        ctx.outputs[key] = read_csv_columns(ctx.run_dirs["sim"] / f"{name}_trace.csv", TRACE)
    out: dict[str, FloatArray] = ctx.outputs[key]
    return out


def _time_fraction(mask: NDArray[np.bool_]) -> float:
    """Fraction of time spent where `mask` holds. Samples are uniform in time;
    the initial condition (index 0) is not a time interval and is excluded."""
    return float(np.mean(mask[1:]))


@check("ACTUATOR")
def actuator(ctx: CheckContext) -> Outcome:
    for name in SCENARIOS:
        pump = read_csv_columns(ctx.run_dirs["sim"] / f"{name}_pump.csv", PUMP)
        req = pump["requested_mU_per_min"]
        if np.any(req < 0) or np.any(req > U_MAX):
            return Outcome(False, f"{name}: requested rate outside [0, {U_MAX:g}] mU/min")
    return Outcome(True)


@check("HYPO")
def hypoglycaemia(ctx: CheckContext) -> Outcome:
    low = min(float(_trace(ctx, n)["plasma_glucose_mg_dL"].min()) for n in SCENARIOS)
    return Outcome(low >= HYPO_LIMIT, f"glucose fell to {low:.0f} mg/dL", low, HYPO_LIMIT)


@check("TIME_IN_RANGE")
def time_in_range(ctx: CheckContext) -> Outcome:
    g = _trace(ctx, "day")["plasma_glucose_mg_dL"]
    tir = _time_fraction((g >= TIR_RANGE[0]) & (g <= TIR_RANGE[1]))
    return Outcome(tir >= TIR_MIN, "time in range 70-180 mg/dL below 70 %", tir, TIR_MIN)


@check("TIME_ABOVE_250")
def time_above_250(ctx: CheckContext) -> Outcome:
    g = _trace(ctx, "day")["plasma_glucose_mg_dL"]
    tar = _time_fraction(g > 250.0)
    return Outcome(tar <= TAR250_MAX, "time above 250 mg/dL above 5 %", tar, TAR250_MAX)


@check("CORRECTION_TIME")
def correction_time(ctx: CheckContext) -> Outcome:
    tr = _trace(ctx, "correction")
    below = np.flatnonzero(tr["plasma_glucose_mg_dL"] < 180.0)
    when = float(tr["t_min"][below[0]]) if below.size else math.inf
    return Outcome(
        when <= CORRECTION_LIMIT_MIN,
        "hyperglycaemia not corrected within 2 h",
        when if math.isfinite(when) else None,
        CORRECTION_LIMIT_MIN,
    )


@check("SETTLING")
def settling(ctx: CheckContext) -> Outcome:
    """After the correction, glucose must settle: no sustained oscillation and a
    sensible resting level. A marginally stable loop fails here."""
    tr = _trace(ctx, "correction")
    late = tr["plasma_glucose_mg_dL"][tr["t_min"] >= CORRECTION_MIN - SETTLE_WINDOW_MIN]
    p2p = float(np.ptp(late))
    mean = float(late.mean())
    if not SETTLE_MEAN[0] <= mean <= SETTLE_MEAN[1]:
        return Outcome(False, f"settled at {mean:.0f} mg/dL, outside {SETTLE_MEAN}")
    return Outcome(p2p <= SETTLE_P2P_MAX, "sustained oscillation", p2p, SETTLE_P2P_MAX)


@register
class GlucoseTask(TaskPlugin):
    task_id = "t03-glucose-closed-loop"
    version = "1.0.0"
    timeout_s = 600.0  # whole session; per-step deadlines are much tighter

    def languages(self) -> tuple[str, ...]:
        return ("python", "octave")

    def make_case(self, seed: int) -> GlucoseCase:
        return make_case(seed)

    def write_inputs(self, case: GlucoseCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}",)

    def execute(
        self, case: Any, staged: Path, runner: Runner, case_dir: Path, lang: str
    ) -> tuple[Execution, dict[str, Path]]:
        sim_dir = case_dir / "sim"  # grader-owned: the controller cannot write here
        sim_dir.mkdir(parents=True, exist_ok=True)
        run_dirs = {"sim": sim_dir}
        sensor_seeds = [int(s.integers(0, 2**63)) for s in streams(case.seed ^ 0x5E750, 2)]
        for scenario, sensor_seed in zip(case.scenarios, sensor_seeds, strict=True):
            out = case_dir / f"controller-{scenario.name}"
            session = runner.start(
                self.spec(staged, case_dir / "input", out, lang, RunPlan(scenario.name)),
                FIRST_STEP_TIMEOUT_S,
            )

            def request(message: dict[str, Any], first: bool, s: Any = session) -> dict[str, Any]:
                s.step_timeout_s = FIRST_STEP_TIMEOUT_S if first else STEP_TIMEOUT_S
                reply: dict[str, Any] = s.request(message)
                return reply

            try:
                trace, pump = run_scenario(case, scenario, request, sensor_seed)
            except ProtocolError as exc:
                return Execution(False, f"{scenario.name}: {exc}"), run_dirs
            finally:
                session.close()
            # Written only after the controller's process group is gone, into a
            # grader-owned directory the sandbox user cannot write.
            write_sim(sim_dir, scenario.name, trace, pump)
        return Execution(True), run_dirs

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t03",
            [
                actuator,
                CheckGroup("safety", [hypoglycaemia]),
                CheckGroup("performance", [time_in_range, time_above_250]),
                CheckGroup("dynamics", [correction_time, settling]),
            ],
        )
