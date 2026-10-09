"""Linear two-compartment pharmacokinetics with zero-order (constant-rate) infusions.

State: drug amounts A1 (central) and A2 (peripheral), mg. Time in hours.

    dA1/dt = -(k10 + k12) A1 + k21 A2 + R(t)
    dA2/dt =   k12 A1        - k21 A2

with k10 = CL/V1, k12 = Q/V1, k21 = Q/V2 (1/h) and plasma concentration
C = A1/V1 (mg/L). R(t) is piecewise constant, so on every interval between
infusion start/stop times the system is linear time-invariant with a constant
input, and is solved *exactly* with the matrix exponential of the augmented
system

    d/dt [x; 1] = [[K, r], [0, 0]] [x; 1],   x(t+h) = E11 x(t) + E12,  E = expm(M h)

so there is no time-stepping error at all. This is the grader's ground truth
and the yardstick for a submission's own solver.

Linearity (superposition) also makes steady state exact: one dosing period is
the affine map x -> Phi x + b, with fixed point x_ss = (I - Phi)^-1 b, and every
steady-state concentration scales linearly with dose.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

FloatArray = NDArray[np.float64]
PARAM_NAMES = ("CL", "V1", "Q", "V2")  # L/h, L, L/h, L


@dataclass(frozen=True)
class Infusion:
    start_h: float
    duration_h: float
    amount_mg: float

    @property
    def rate_mg_per_h(self) -> float:
        return self.amount_mg / self.duration_h


def rate_matrix(p: FloatArray) -> FloatArray:
    cl, v1, q, v2 = p
    k10, k12, k21 = cl / v1, q / v1, q / v2
    return np.array([[-(k10 + k12), k21], [k12, -k21]])


def half_lives(p: FloatArray) -> tuple[float, float]:
    """Distribution (alpha) and terminal (beta) half-lives in h, from the
    eigenvalues -alpha < -beta < 0 of the rate matrix."""
    eig = np.sort(-np.linalg.eigvals(rate_matrix(p)).real)[::-1]  # alpha > beta
    return float(np.log(2) / eig[0]), float(np.log(2) / eig[1])


def _propagate(k: FloatArray, x: FloatArray, dt: float, rate: float) -> FloatArray:
    if dt <= 0.0:
        return x
    m = np.zeros((3, 3))
    m[:2, :2] = k
    m[0, 2] = rate
    e = expm(m * dt)
    out: FloatArray = e[:2, :2] @ x + e[:2, 2]
    return out


def concentrations(p: FloatArray, doses: list[Infusion], t: FloatArray) -> FloatArray:
    """Exact plasma concentration (mg/L) at sorted times `t` (h), starting drug-free at 0."""
    k = rate_matrix(p)
    events = {0.0, *t.tolist()}
    for d in doses:
        events |= {d.start_h, d.start_h + d.duration_h}
    grid = sorted(e for e in events if e >= 0.0)

    def rate(at: float) -> float:
        return sum(d.rate_mg_per_h for d in doses if d.start_h <= at < d.start_h + d.duration_h)

    x = np.zeros(2)
    values: dict[float, float] = {0.0: 0.0}
    for a, b in itertools.pairwise(grid):
        x = _propagate(k, x, b - a, rate(0.5 * (a + b)))
        values[b] = x[0] / p[1]
    return np.array([values[ti] for ti in t.tolist()])


@dataclass(frozen=True)
class SteadyState:
    peak_mg_per_l: float
    trough_mg_per_l: float


def steady_state(
    p: FloatArray, dose_mg: float, tau_h: float, infusion_h: float, n_grid: int = 2001
) -> SteadyState:
    """Peak and trough over one steady-state interval of repeated infusions."""
    k = rate_matrix(p)
    r = dose_mg / infusion_h
    b = _propagate(k, _propagate(k, np.zeros(2), infusion_h, r), tau_h - infusion_h, 0.0)
    phi = expm(k * tau_h)
    x0 = np.linalg.solve(np.eye(2) - phi, b)
    x_end_inf = _propagate(k, x0, infusion_h, r)
    trough = x0[0] / p[1]
    # The maximum of the central compartment lies inside or at the end of the
    # infusion; evaluate the end point exactly and a dense grid for safety.
    ts = np.linspace(0.0, tau_h, n_grid)
    grid = [
        _propagate(k, x0, t, r)[0]
        if t <= infusion_h
        else _propagate(k, x_end_inf, t - infusion_h, 0.0)[0]
        for t in ts
    ]
    peak = max(*grid, x_end_inf[0]) / p[1]
    return SteadyState(float(peak), float(trough))


@dataclass(frozen=True)
class Regimen:
    tau_h: float
    dose_mg: float
    daily_mg: float


def optimal_regimen(
    p: FloatArray, window: tuple[float, float], taus: tuple[float, ...], infusion_h: float
) -> Regimen | None:
    """Smallest daily dose keeping steady state inside `window` (mg/L).

    By linearity, for each interval tau the feasible doses form the interval
    [lo / trough_per_mg, hi / peak_per_mg]; the cheapest feasible choice is its
    lower end, and the optimum is the cheapest over the allowed taus. The
    constrained optimisation is therefore exact, with no iterative solver.
    """
    best: Regimen | None = None
    for tau in taus:
        unit = steady_state(p, 1.0, tau, infusion_h)
        d_min = window[0] / unit.trough_mg_per_l
        d_max = window[1] / unit.peak_mg_per_l
        if d_min <= d_max:
            daily = d_min * 24.0 / tau
            if best is None or daily < best.daily_mg:
                best = Regimen(tau, d_min, daily)
    return best
