"""Verification of the McSharry model implementation (not validation of physiology)."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bioengbench.models.ecg import (
    STANDARD_WAVES,
    RhythmParams,
    forcing,
    integrate_z,
    r_event_times,
    simulate,
    wrap_angle,
)


def _u(t: np.ndarray) -> np.ndarray:
    """Smooth forcing with a sharp QRS-like bump (width 15 ms)."""
    return np.sin(3.0 * t) + 30.0 * np.exp(-0.5 * ((t - 1.0) / 0.015) ** 2)


def _reference(t_end: float) -> float:
    sol = solve_ivp(
        lambda t, z: -z + _u(np.array([t])),
        (0.0, t_end),
        [0.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        max_step=1e-3,
    )
    return float(sol.y[0, -1])


def test_first_order_hold_integrator_is_second_order() -> None:
    """Observed order p = log2(e(h) / e(h/2)) from three refinements must be ~2."""
    t_end = 2.0
    ref = _reference(t_end)
    errors = []
    for n in (500, 1000, 2000, 4000):
        h = t_end / n
        t = np.arange(n + 1) * h
        errors.append(abs(integrate_z(_u(t), h)[-1] - ref))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.9 < p < 2.1 for p in orders), orders


def test_integrator_matches_exact_solution_for_constant_forcing() -> None:
    h = 0.01
    z = integrate_z(np.ones(501), h)
    t = np.arange(501) * h
    np.testing.assert_allclose(z, 1.0 - np.exp(-t), atol=1e-12)


def test_r_events_of_uniform_rotation() -> None:
    f = 1.25  # Hz
    t = np.linspace(0, 10, 200_001)
    theta = -np.pi + 2 * np.pi * f * t
    events = r_event_times(t, theta)
    expected = (np.arange(len(events)) + 0.5) / f  # theta = 0 at t = 0.4, 1.2, ...
    np.testing.assert_allclose(events, expected, atol=1e-6)


def test_wrap_angle_range() -> None:
    a = np.linspace(-20, 20, 10_001)
    w = wrap_angle(a)
    assert np.all(w > -np.pi - 1e-12) and np.all(w <= np.pi + 1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)


def test_forcing_is_odd_around_each_wave_centre() -> None:
    d = np.linspace(-0.05, 0.05, 101)
    f = forcing(d, {"R": STANDARD_WAVES["R"]})
    np.testing.assert_allclose(f, -f[::-1], atol=1e-12)


@pytest.fixture(scope="module")
def sim():
    rhythm = RhythmParams(75.0, 0.25, 0.04, 0.02, 0.01, 0.0, 1.0)
    return simulate(rhythm, STANDARD_WAVES, 60.0, 360.0, np.random.default_rng(0))


def test_simulation_has_expected_beats_and_morphology(sim) -> None:
    n = len(sim.r_times)
    assert 70 <= n <= 80  # 75 bpm for 60 s
    idx = np.round(sim.r_times * 360).astype(int)
    idx = idx[(idx > 100) & (idx < len(sim.z) - 100)]
    r = np.median(sim.z[idx])
    t_wave = np.median([sim.z[i + 60 : i + 110].max() for i in idx])  # 170-300 ms after R
    q_s = np.median([sim.z[i - 15 : i + 15].min() for i in idx])
    assert r > 0 and 0.15 < t_wave / r < 0.6 and q_s < 0


def test_simulation_is_deterministic() -> None:
    rhythm = RhythmParams(75.0, 0.25, 0.04, 0.02, 0.01, 0.0, 1.0)
    a = simulate(rhythm, STANDARD_WAVES, 10.0, 360.0, np.random.default_rng(5))
    b = simulate(rhythm, STANDARD_WAVES, 10.0, 360.0, np.random.default_rng(5))
    assert np.array_equal(a.z, b.z)
