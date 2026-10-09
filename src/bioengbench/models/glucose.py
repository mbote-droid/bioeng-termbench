"""Bergman minimal model of glucose-insulin dynamics, type 1 diabetes form,
with meal absorption and a continuous glucose monitor (CGM).

Time in minutes. Glucose G in mg/dL, remote insulin action X in 1/min, plasma
insulin I in mU/L, pump rate u in mU/min (1 U/h = 16.67 mU/min).

    dG/dt  = -p1 (G - Gb) - (X + (a(t) - 1) max(X, 0)) G - k_ex(t) G + Ra(t) / VG
    dX/dt  = -p2 X + p3 (I - Ib)
    dI/dt  = -n I + u(t) / VI
    dGi/dt = (G - Gi) / tau_s          (interstitial glucose, what the CGM sees)

Ib = ub / (n VI) is the plasma insulin produced by the basal rate ub, so with
u = ub the patient rests at G = Gb. Glucose effectiveness p1 is small in type 1
diabetes, so without insulin feedback glucose corrects only slowly. Insulin
sensitivity S_I = p3 / p2.

a(t) >= 1 and k_ex(t) >= 0 model exercise (see Exercise); both are 1 and 0 at
rest. Exercise amplifies the action of insulin above basal (X > 0) only; it
does not amplify the hyperglycaemic effect of an insulin deficit (X < 0).

Meals: Ra(t) = 1000 * CHO * f * s / tmax^2 * exp(-s / tmax), s = t - t_meal
(mg/min), a gamma-shaped absorption with bioavailability f = 0.8.

Integration: classical RK4 with a 0.5 min step inside each 5 min pump period,
the insulin rate held constant over the period (zero-order hold, as a pump
does). RK4's local error on these time constants (>= 5 min) is far below a
milligram per decilitre; tests confirm it against a tight LSODA solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
SAMPLE_MIN = 5.0
SUBSTEP_MIN = 0.5
BIOAVAILABILITY = 0.8


@dataclass(frozen=True)
class Patient:
    body_mass_kg: float
    p1: float  # 1/min
    p2: float  # 1/min
    p3: float  # 1/min^2 per mU/L
    n: float  # 1/min
    gb: float  # mg/dL, glucose at the basal rate
    ub: float  # mU/min, basal rate (known to the pump)
    tau_s: float  # min, plasma-to-interstitial lag

    @property
    def vi(self) -> float:  # L
        return 0.12 * self.body_mass_kg

    @property
    def vg(self) -> float:  # dL
        return 1.6 * self.body_mass_kg

    @property
    def ib(self) -> float:  # mU/L
        return self.ub / (self.n * self.vi)

    @property
    def insulin_sensitivity(self) -> float:
        return self.p3 / self.p2


@dataclass(frozen=True)
class Meal:
    t_min: float
    carbs_g: float
    tmax_min: float


def meal_rate(t: float, meals: tuple[Meal, ...]) -> float:
    """Glucose appearance from all meals at time t, mg/min."""
    total = 0.0
    for m in meals:
        s = t - m.t_min
        if s > 0:
            shape = s / m.tmax_min**2 * np.exp(-s / m.tmax_min)
            total += 1000.0 * m.carbs_g * BIOAVAILABILITY * shape
    return total


@dataclass(frozen=True)
class Exercise:
    """Moderate aerobic exercise, simplified from Breton (2008) and Roy & Parker
    (2007): during the bout, insulin-independent uptake k_uptake * G and an
    action of insulin above basal amplified by (1 + gain); afterwards the amplification decays
    with time constant tau_post_min (post-exercise sensitivity, the classic
    cause of delayed hypoglycaemia)."""

    start_min: float
    duration_min: float
    k_uptake: float  # 1/min
    gain: float  # dimensionless
    tau_post_min: float = 90.0

    def effect(self, t: float) -> tuple[float, float]:
        """(extra uptake rate 1/min, insulin action multiplier) at time t."""
        s = t - self.start_min
        if s < 0:
            return 0.0, 1.0
        if s < self.duration_min:
            return self.k_uptake, 1.0 + self.gain
        return 0.0, 1.0 + self.gain * float(np.exp(-(s - self.duration_min) / self.tau_post_min))


@dataclass(frozen=True)
class Disturbances:
    meals: tuple[Meal, ...] = ()
    exercise: tuple[Exercise, ...] = ()


def derivative(p: Patient, x: FloatArray, t: float, u: float, d: Disturbances) -> FloatArray:
    g, xa, ins, gi = x
    uptake, action = 0.0, 1.0
    for e in d.exercise:
        k, a = e.effect(t)
        uptake += k
        action *= a
    return np.array(
        [
            -p.p1 * (g - p.gb)
            - (xa + (action - 1.0) * max(xa, 0.0)) * g
            - uptake * g
            + meal_rate(t, d.meals) / p.vg,
            -p.p2 * xa + p.p3 * (ins - p.ib),
            -p.n * ins + u / p.vi,
            (g - gi) / p.tau_s,
        ]
    )


def advance(
    p: Patient,
    x: FloatArray,
    t: float,
    u: float,
    d: Disturbances,
    *,
    dt: float = SAMPLE_MIN,
) -> tuple[FloatArray, FloatArray]:
    """RK4 over one pump period with constant u. Returns the new state and the
    plasma glucose at every substep (for time-in-range statistics)."""
    h = SUBSTEP_MIN
    glucose = []
    for k in range(round(dt / h)):
        tk = t + k * h
        k1 = derivative(p, x, tk, u, d)
        k2 = derivative(p, x + 0.5 * h * k1, tk + 0.5 * h, u, d)
        k3 = derivative(p, x + 0.5 * h * k2, tk + 0.5 * h, u, d)
        k4 = derivative(p, x + h * k3, tk + h, u, d)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        glucose.append(x[0])
    return x, np.array(glucose)


def initial_state(p: Patient, g0: float) -> FloatArray:
    """Insulin at basal steady state, glucose (plasma and interstitial) at g0."""
    return np.array([g0, 0.0, p.ib, g0])


@dataclass(frozen=True)
class SensorFault:
    start_min: float
    end_min: float
    kind: str  # "dropout" (no reading) or "offset" (artefact added to reading)
    offset_mg_dl: float = 0.0


@dataclass
class Cgm:
    """CGM: interstitial glucose + AR(1) calibration drift + proportional white
    noise, plus scheduled faults. Noise draws come from a dedicated stream so
    the reading sequence does not depend on the controller's actions."""

    rng: np.random.Generator
    faults: tuple[SensorFault, ...] = ()
    _drift: float = field(default=0.0, init=False)

    def read(self, gi: float, t: float) -> float | None:
        self._drift = 0.7 * self._drift + self.rng.normal(0.0, 3.0)
        white = self.rng.normal(0.0, 0.03 * gi)
        value: float | None = gi + self._drift + white
        for f in self.faults:
            if f.start_min <= t < f.end_min:
                value = None if f.kind == "dropout" or value is None else value + f.offset_mg_dl
        if value is not None:
            value = float(np.clip(value, 40.0, 400.0))  # sensor reporting range
        return value
