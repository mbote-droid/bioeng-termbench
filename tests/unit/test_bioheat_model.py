"""Verification of the bioheat reference solver (t04): exact solutions,
observed order of accuracy, discrete energy conservation, Picard coupling."""

import math

import numpy as np
import pytest
from scipy.special import erfc

from bioengbench.models.bioheat import (
    Problem,
    center_value,
    conduction_only_center_rise,
    shell_power_fraction,
    simulate,
    stasis,
)

PROB = Problem(sigma_m=0.012)


def test_source_fractions_sum_to_one() -> None:
    faces = np.linspace(0.0, 0.2, 2001)
    frac = shell_power_fraction(faces[:-1], faces[1:], 0.012)
    assert frac.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(frac >= 0)


def test_center_extrapolation_is_exact_for_even_quadratics() -> None:
    h = 0.1
    r = np.array([h / 2, 3 * h / 2])
    assert center_value(5.0 - 2.0 * r**2) == pytest.approx(5.0)


def test_conduction_matches_the_exact_gaussian_kernel_solution() -> None:
    """No perfusion, short time (far boundary not yet reached)."""
    res = simulate(
        PROB, power_w=5.0, heating_s=120.0, perfusion_per_s=0.0, end_s=120.0, nr=600, dt=0.5
    )
    exact = 37.0 + conduction_only_center_rise(PROB, 5.0, 120.0)
    assert res.t_center_peak_c == pytest.approx(exact, abs=2e-3)


def test_long_time_solution_matches_steady_pennes() -> None:
    """Constant perfusion (power low enough that damage is negligible): the
    steady state is the Gaussian source convolved with the screened Green's
    function exp(-m r) / (4 pi k r), m = sqrt(w / k)."""
    p, omega = 1.0, 0.006
    res = simulate(
        PROB, power_w=p, heating_s=4000.0, perfusion_per_s=omega, end_s=4000.0, nr=600, dt=10.0
    )
    assert res.omega_end.max() < 2e-3  # so perfusion is within 0.2 % of constant
    k = PROB.tissue.conductivity
    m = math.sqrt(omega * PROB.tissue.blood_heat_capacity / k)
    s = PROB.sigma_m
    r = res.r_m[20:300]
    exact = (
        p
        / (8 * math.pi * k * r)
        * math.exp(m * m * s * s / 4)
        * (np.exp(-m * r) * erfc(m * s / 2 - r / s) - np.exp(m * r) * erfc(m * s / 2 + r / s))
    )
    np.testing.assert_allclose(res.theta_end[20:300], exact, rtol=3e-3, atol=1e-5)


def test_energy_is_conserved_to_round_off_at_any_resolution() -> None:
    for nr, dt in ((40, 10.0), (200, 2.0)):
        b = simulate(
            PROB, power_w=9.0, heating_s=600.0, perfusion_per_s=0.005, end_s=900.0, nr=nr, dt=dt
        ).budget
        assert abs(b.residual) < 1e-9 * b.deposited
        assert b.deposited == pytest.approx(9.0 * 600.0, rel=1e-9)
        assert min(b.stored, b.perfusion, b.boundary) >= 0.0


def _orders(picard: int) -> dict[str, float]:
    levels = [(100, 4.0), (200, 2.0), (400, 1.0)]
    runs = [
        simulate(
            PROB,
            power_w=9.0,
            heating_s=600.0,
            perfusion_per_s=0.005,
            end_s=900.0,
            nr=n,
            dt=d,
            picard=picard,
        )
        for n, d in levels
    ]
    out = {}
    for key in ("t_center_peak_c", "energy_at_ton_j"):
        v = [getattr(x, key) for x in runs]
        out[key] = math.log2(abs(v[0] - v[1]) / abs(v[1] - v[2]))
    return out


@pytest.mark.slow
def test_second_order_in_space_and_time_with_picard_coupling() -> None:
    assert all(p > 1.8 for p in _orders(3).values()), _orders(3)


@pytest.mark.slow
def test_lagged_perfusion_is_only_first_order() -> None:
    """The negative control behind t04's CONVERGENCE check."""
    assert all(0.8 < p < 1.3 for p in _orders(1).values()), _orders(1)


def test_vascular_collapse_enlarges_the_lesion() -> None:
    """Hot tissue stops being cooled, so the lesion is larger than with
    constant perfusion (the nonlinearity matters for planning)."""
    kw = dict(power_w=9.0, heating_s=600.0, perfusion_per_s=0.005, end_s=900.0, nr=200, dt=2.0)
    res = simulate(PROB, **kw)
    assert stasis(np.array([60.0 - 37.0]))[0] == pytest.approx(0.5)
    assert stasis(np.array([0.0]))[0] > 0.999
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("bioengbench.models.bioheat.stasis", np.ones_like)
        constant = simulate(PROB, **kw)
    assert res.ablation_radius_mm > constant.ablation_radius_mm + 0.3
    assert res.ablation_radius_mm > 8.0


def test_more_perfusion_means_a_smaller_cooler_lesion() -> None:
    a, b = (
        simulate(PROB, power_w=9.0, heating_s=600.0, perfusion_per_s=w, end_s=900.0, nr=100, dt=4.0)
        for w in (0.004, 0.006)
    )
    assert b.ablation_radius_mm < a.ablation_radius_mm
    assert b.t_center_peak_c < a.t_center_peak_c


def test_time_grid_must_hit_the_power_switch() -> None:
    with pytest.raises(ValueError, match="multiples"):
        simulate(PROB, power_w=1.0, heating_s=601.0, perfusion_per_s=0.0, end_s=900.0, dt=2.0)
