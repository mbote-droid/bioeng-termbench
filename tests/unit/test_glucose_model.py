"""Verification of the glucose-insulin plant used by t03."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bioengbench.models.glucose import (
    SAMPLE_MIN,
    Cgm,
    Disturbances,
    Exercise,
    Meal,
    Patient,
    SensorFault,
    advance,
    derivative,
    initial_state,
    meal_rate,
)

PATIENT = Patient(
    body_mass_kg=70.0, p1=0.005, p2=0.025, p3=0.025 * 5e-4, n=0.12, gb=120.0, ub=17.0, tau_s=8.0
)
DIST = Disturbances(
    meals=(Meal(30.0, 50.0, 50.0), Meal(300.0, 40.0, 45.0)),
    exercise=(Exercise(400.0, 45.0, k_uptake=0.006, gain=0.8),),
)


def _rk4(u_of_t, duration, x0, d=DIST):
    x, out = x0, [x0[0]]
    for k in range(round(duration / SAMPLE_MIN)):
        t = k * SAMPLE_MIN
        x, g = advance(PATIENT, x, t, u_of_t(t), d)
        out.extend(g.tolist())
    return x, np.array(out)


def test_basal_rate_is_an_equilibrium_at_gb() -> None:
    x0 = initial_state(PATIENT, PATIENT.gb)
    np.testing.assert_allclose(
        derivative(PATIENT, x0, 0.0, PATIENT.ub, Disturbances()), 0.0, atol=1e-12
    )
    x, _ = _rk4(lambda t: PATIENT.ub, 600.0, x0, Disturbances())
    np.testing.assert_allclose(x, x0, atol=1e-9)


def test_rk4_matches_a_tight_lsoda_solution() -> None:
    """Zero-order-hold input is piecewise constant; integrate LSODA period by
    period so its own error is far below RK4's."""
    u_of_t = lambda t: PATIENT.ub * (1.0 + 2.0 * np.sin(t / 97.0) ** 2)  # noqa: E731
    x0 = initial_state(PATIENT, 180.0)
    x_rk, _ = _rk4(u_of_t, 720.0, x0)
    x = x0.copy()
    for k in range(round(720.0 / SAMPLE_MIN)):
        t = k * SAMPLE_MIN
        u = u_of_t(t)
        sol = solve_ivp(
            lambda tt, y, uu=u: derivative(PATIENT, y, tt, uu, DIST),
            (t, t + SAMPLE_MIN),
            x,
            method="LSODA",
            rtol=1e-11,
            atol=1e-11,
        )
        x = sol.y[:, -1]
    assert abs(x_rk[0] - x[0]) < 0.05  # mg/dL
    np.testing.assert_allclose(x_rk, x, rtol=1e-4, atol=1e-6)


def test_meal_absorbs_its_bioavailable_carbohydrate() -> None:
    t = np.arange(0.0, 2000.0, 0.1)
    total = np.trapezoid([meal_rate(ti, (Meal(0.0, 50.0, 50.0),)) for ti in t], t)
    assert total == pytest.approx(50.0 * 1000.0 * 0.8, rel=1e-3)


def test_meal_raises_and_insulin_lowers_glucose() -> None:
    x0 = initial_state(PATIENT, PATIENT.gb)
    meal = Disturbances(meals=(Meal(0.0, 50.0, 50.0),))
    _, g_basal = _rk4(lambda t: PATIENT.ub, 240.0, x0, meal)
    _, g_more = _rk4(lambda t: 3 * PATIENT.ub, 240.0, x0, meal)
    assert g_basal.max() > PATIENT.gb + 50
    assert np.all(g_more[1:] < g_basal[1:])


def test_exercise_lowers_glucose_and_raises_insulin_action_afterwards() -> None:
    ex = Exercise(100.0, 45.0, k_uptake=0.006, gain=0.8)
    assert ex.effect(50.0) == (0.0, 1.0)
    assert ex.effect(120.0) == (0.006, 1.8)
    k, a = ex.effect(145.0 + 90.0)
    assert k == 0.0
    assert a == pytest.approx(1.0 + 0.8 / np.e)
    x0 = initial_state(PATIENT, PATIENT.gb)
    _, g_rest = _rk4(lambda t: PATIENT.ub, 300.0, x0, Disturbances())
    _, g_ex = _rk4(lambda t: PATIENT.ub, 300.0, x0, Disturbances(exercise=(ex,)))
    assert g_ex.min() < g_rest.min() - 20


def test_exercise_does_not_amplify_an_insulin_deficit() -> None:
    """With insulin suspended (X < 0) exercise must not make glucose rise faster."""
    x0 = initial_state(PATIENT, PATIENT.gb)
    ex = Disturbances(exercise=(Exercise(0.0, 45.0, k_uptake=0.0, gain=1.0),))
    _, g_ex = _rk4(lambda t: 0.0, 45.0, x0, ex)
    _, g_rest = _rk4(lambda t: 0.0, 45.0, x0, Disturbances())
    np.testing.assert_allclose(g_ex, g_rest)


def test_interstitial_glucose_lags_plasma() -> None:
    x0 = initial_state(PATIENT, PATIENT.gb)
    x, _ = _rk4(lambda t: PATIENT.ub, 60.0, x0, Disturbances(meals=(Meal(0.0, 60.0, 40.0),)))
    assert x[3] < x[0]


def test_cgm_noise_is_reproducible_and_faults_apply() -> None:
    faults = (SensorFault(10, 20, "dropout"), SensorFault(30, 40, "offset", 80.0))
    reads = []
    for _ in range(2):
        cgm = Cgm(np.random.default_rng(5), faults)
        reads.append([cgm.read(120.0, t) for t in range(0, 50, 5)])
    assert reads[0] == reads[1]
    assert reads[0][2] is None and reads[0][3] is None
    assert reads[0][6] > 170.0
    assert all(40.0 <= r <= 400.0 for r in reads[0] if r is not None)
