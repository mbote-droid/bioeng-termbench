"""McSharry dynamical ECG model (McSharry et al., IEEE TBME 50(3), 2003).

The model is a trajectory on a limit cycle in (x, y) plus a vertical
coordinate z that carries the ECG voltage:

    dx/dt = alpha*x - omega(t)*y
    dy/dt = alpha*y + omega(t)*x,          alpha = 1 - sqrt(x^2 + y^2)
    dz/dt = -sum_i a_i * dth_i * exp(-dth_i^2 / (2 b_i^2)) - (z - z0)

with theta = atan2(y, x), dth_i = (theta - theta_i) wrapped to (-pi, pi],
and one (theta_i, a_i, b_i) triple per wave i in {P, Q, R, S, T}.

Solution strategy (and why it is exact where it can be):
* Starting on the limit cycle (r = 1) gives alpha = 0 for all time, so the
  phase is theta(t) = theta(0) + integral of omega. We integrate omega with
  the trapezoid rule on a fine grid instead of stepping x and y.
* The z equation is then linear and first order: dz/dt = -z + u(t), with
  u = z0 + F(theta(t)). We discretise it with the exact first-order-hold
  exponential integrator (u linear inside each step):

      z[n+1] = e^-h z[n] + (1 - e^-h) u[n] + (h - 1 + e^-h)/h * (u[n+1] - u[n])

  which is second-order accurate (verified in tests by Richardson
  extrapolation against a tight-tolerance RK45 solve) and runs as a single
  IIR filter in C via scipy.signal.lfilter.

Beat timing follows physiology: a mean RR interval modulated by respiratory
sinus arrhythmia (respiration frequency) and Mayer waves (about 0.1 Hz), plus
a smoothed autoregressive component. R events are the phase crossings of
theta_R = 0.

Units: time in s, phase in rad, ECG voltage in mV after scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter, lfiltic

FloatArray = NDArray[np.float64]
TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Wave:
    theta: float  # angular position on the cycle, rad (R wave at 0)
    a: float  # forcing amplitude, dimensionless model units
    b: float  # angular width, rad


# Standard parameters from McSharry et al. (2003), Table 1.
STANDARD_WAVES: dict[str, Wave] = {
    "P": Wave(-np.pi / 3, 1.2, 0.25),
    "Q": Wave(-np.pi / 12, -5.0, 0.1),
    "R": Wave(0.0, 30.0, 0.1),
    "S": Wave(np.pi / 12, -7.5, 0.1),
    "T": Wave(np.pi / 2, 0.75, 0.4),
}


@dataclass(frozen=True)
class RhythmParams:
    mean_hr_bpm: float
    resp_hz: float  # respiration frequency, drives sinus arrhythmia
    rsa_frac: float  # RR modulation depth from respiration (fraction of mean RR)
    mayer_frac: float  # RR modulation depth at the Mayer-wave frequency
    ar_sd_frac: float  # SD of the beat-to-beat autoregressive component (fraction)
    phase_rsa: float
    phase_mayer: float

    MAYER_HZ = 0.1


def wrap_angle(a: FloatArray) -> FloatArray:
    """Wrap to (-pi, pi]."""
    return np.asarray(np.pi - np.mod(np.pi - a, TWO_PI), dtype=np.float64)


def rr_function(rhythm: RhythmParams, duration: float, rng: np.random.Generator) -> CubicSpline:
    """Continuous RR(t) in seconds: deterministic oscillations plus a smooth AR(1) term.

    The AR(1) innovations are drawn once per second and joined by a cubic spline,
    so RR(t) is smooth (no step changes in heart rate).
    """
    mean_rr = 60.0 / rhythm.mean_hr_bpm
    knots = np.arange(-2.0, duration + 3.0, 1.0)
    ar = np.zeros_like(knots)
    for k in range(1, len(knots)):
        ar[k] = 0.6 * ar[k - 1] + rng.normal(0.0, rhythm.ar_sd_frac)
    rr = mean_rr * (
        1.0
        + rhythm.rsa_frac * np.sin(TWO_PI * rhythm.resp_hz * knots + rhythm.phase_rsa)
        + rhythm.mayer_frac * np.sin(TWO_PI * rhythm.MAYER_HZ * knots + rhythm.phase_mayer)
        + ar
    )
    return CubicSpline(knots, rr)


def phase(t: FloatArray, rr: CubicSpline, theta0: float) -> FloatArray:
    """Unwrapped phase theta(t) = theta0 + integral of 2*pi / RR(t) dt (trapezoid rule)."""
    omega = TWO_PI / rr(t)
    increments = 0.5 * (omega[1:] + omega[:-1]) * np.diff(t)
    return np.asarray(theta0 + np.concatenate(([0.0], np.cumsum(increments))), dtype=np.float64)


def forcing(theta: FloatArray, waves: dict[str, Wave]) -> FloatArray:
    """F(theta) = -sum_i a_i dth_i exp(-dth_i^2 / 2 b_i^2)."""
    f = np.zeros_like(theta)
    for w in waves.values():
        d = wrap_angle(theta - w.theta)
        f -= w.a * d * np.exp(-0.5 * (d / w.b) ** 2)
    return f


def integrate_z(u: FloatArray, h: float) -> FloatArray:
    """Solve dz/dt = -z + u(t), z(0) = 0, with the exact first-order-hold step.

    As an IIR filter: z[n] = e^-h z[n-1] + beta0 u[n] + beta1 u[n-1]. The
    filter state is initialised from (z[0] = 0, u[0]) so the first step is
    z[1] = beta0 u[1] + beta1 u[0]; without that, lfilter would assume
    u[-1] = 0 and inject a spurious beta0 * u[0] at t = 0.
    """
    decay = np.exp(-h)
    beta0 = (h - 1.0 + decay) / h
    beta1 = (1.0 - decay) - beta0
    b, a = [beta0, beta1], [1.0, -decay]
    z = np.zeros_like(u, dtype=np.float64)
    zi = lfiltic(b, a, y=[0.0], x=[u[0]])
    z[1:], _ = lfilter(b, a, u[1:], zi=zi)
    return z


def r_event_times(t: FloatArray, theta: FloatArray, theta_r: float = 0.0) -> FloatArray:
    """Times where the unwrapped phase crosses theta_r + 2*pi*k (linear interpolation)."""
    k_first = int(np.ceil((theta[0] - theta_r) / TWO_PI))
    k_last = int(np.floor((theta[-1] - theta_r) / TWO_PI))
    targets = theta_r + TWO_PI * np.arange(k_first, k_last + 1)
    idx = np.searchsorted(theta, targets)
    idx = idx[(idx > 0) & (idx < len(theta))]
    targets = targets[: len(idx)]
    frac = (targets - theta[idx - 1]) / (theta[idx] - theta[idx - 1])
    return np.asarray(t[idx - 1] + frac * (t[idx] - t[idx - 1]), dtype=np.float64)


@dataclass(frozen=True)
class EcgSimulation:
    t: FloatArray  # output time axis, s
    z: FloatArray  # model voltage, arbitrary units (scale before use)
    r_times: FloatArray  # R-event times, s


def simulate(
    rhythm: RhythmParams,
    waves: dict[str, Wave],
    duration: float,
    fs: float,
    rng: np.random.Generator,
    *,
    oversample: int = 8,
) -> EcgSimulation:
    """Simulate the noise-free model on a grid `oversample` times finer than fs,
    then sample every `oversample`-th point so output samples are exact model
    values (no interpolation)."""
    n_out = round(duration * fs)
    h = 1.0 / (fs * oversample)
    t_fine = np.arange(n_out * oversample, dtype=np.float64) * h
    rr = rr_function(rhythm, duration, rng)
    theta = phase(t_fine, rr, theta0=-np.pi)  # start mid-diastole, away from QRS
    z = integrate_z(forcing(theta, waves), h)
    return EcgSimulation(
        t=t_fine[::oversample],
        z=z[::oversample],
        r_times=r_event_times(t_fine, theta),
    )
