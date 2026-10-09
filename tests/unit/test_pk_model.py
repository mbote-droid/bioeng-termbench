"""Verification of the exact two-compartment solver used as ground truth."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bioengbench.models.pk import (
    Infusion,
    concentrations,
    half_lives,
    optimal_regimen,
    rate_matrix,
    steady_state,
)

P = np.array([4.0, 20.0, 9.0, 35.0])  # CL, V1, Q, V2


def _ode_reference(p, doses, t_end):
    k = rate_matrix(p)

    def rhs(t, x):
        r = sum(d.rate_mg_per_h for d in doses if d.start_h <= t < d.start_h + d.duration_h)
        return k @ x + np.array([r, 0.0])

    events = {*[d.start_h for d in doses], *[d.start_h + d.duration_h for d in doses]}
    bps = sorted({0.0, t_end, *(e for e in events if e < t_end)})
    x = np.zeros(2)
    for a, b in zip(bps[:-1], bps[1:]):  # integrate segment by segment across discontinuities
        x = solve_ivp(rhs, (a, b), x, method="LSODA", rtol=1e-12, atol=1e-12).y[:, -1]
    return x[0] / p[1]


@pytest.mark.parametrize("t_end", [0.5, 1.0, 3.0, 12.5, 30.0])
def test_exact_solution_matches_tight_ode_integration(t_end) -> None:
    doses = [Infusion(0.0, 1.0, 1500.0), Infusion(12.0, 1.0, 1000.0)]
    exact = concentrations(P, doses, np.array([t_end]))[0]
    assert exact == pytest.approx(_ode_reference(P, doses, t_end), rel=1e-9)


def test_terminal_slope_equals_beta_half_life() -> None:
    t = np.array([40.0, 60.0])
    c = concentrations(P, [Infusion(0.0, 1.0, 1000.0)], t)
    slope_half_life = np.log(2) * (t[1] - t[0]) / np.log(c[0] / c[1])
    assert slope_half_life == pytest.approx(half_lives(P)[1], rel=1e-6)


def test_steady_state_equals_limit_of_repeated_dosing() -> None:
    tau, dose = 12.0, 800.0
    doses = [Infusion(i * tau, 1.0, dose) for i in range(60)]
    trough_sim = concentrations(P, doses, np.array([59 * tau]))[0]  # just before dose 60
    assert steady_state(P, dose, tau, 1.0).trough_mg_per_l == pytest.approx(trough_sim, rel=1e-8)


def test_steady_state_is_linear_in_dose() -> None:
    a = steady_state(P, 100.0, 8.0, 1.0)
    b = steady_state(P, 300.0, 8.0, 1.0)
    assert b.peak_mg_per_l == pytest.approx(3 * a.peak_mg_per_l, rel=1e-10)


def test_optimal_regimen_respects_window_and_is_minimal() -> None:
    window = (11.5, 35.0)
    reg = optimal_regimen(P, window, (6.0, 8.0, 12.0, 24.0), 1.0)
    assert reg is not None
    ss = steady_state(P, reg.dose_mg, reg.tau_h, 1.0)
    assert ss.trough_mg_per_l == pytest.approx(window[0], rel=1e-9)  # binding constraint
    assert ss.peak_mg_per_l <= window[1]
    smaller = steady_state(P, 0.99 * reg.dose_mg, reg.tau_h, 1.0)
    assert smaller.trough_mg_per_l < window[0]


def test_mass_balance() -> None:
    """Drug leaves the body only through CL: dose = eliminated + amount remaining."""
    from bioengbench.models.pk import _propagate

    k = rate_matrix(P)
    t = np.linspace(0.0, 48.0, 48_001)
    c = concentrations(P, [Infusion(0.0, 1.0, 1000.0)], t)
    eliminated = np.trapezoid(P[0] * c, t)
    remaining = _propagate(k, _propagate(k, np.zeros(2), 1.0, 1000.0), 47.0, 0.0).sum()
    assert eliminated + remaining == pytest.approx(1000.0, rel=1e-6)
