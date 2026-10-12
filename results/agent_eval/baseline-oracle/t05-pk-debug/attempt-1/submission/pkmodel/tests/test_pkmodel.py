"""Unit tests for pkmodel (run: python -m pytest pkmodel/tests)."""

import numpy as np
import pytest

from pkmodel.grid import output_times
from pkmodel.regimen import Dose
from pkmodel.solver import simulate
from pkmodel.units import Params, clearance_l_per_h

P = Params(cl_l_per_h=5.0, v1_l=20.0, q_l_per_h=10.0, v2_l=40.0)


def test_clearance_conversion():
    assert clearance_l_per_h(100.0) == pytest.approx(6.0)  # 100 mL/min = 6 L/h


def test_grid_endpoints_and_length():
    t = output_times(0.0, 48.0, 0.25)
    assert t.size == 193
    assert t[-1] == 48.0


def test_no_dose_no_drug():
    assert np.all(simulate(P, [], output_times(0, 24, 1)) == 0.0)


def test_infusion_mass_balance_without_elimination():
    p = Params(cl_l_per_h=1e-12, v1_l=20.0, q_l_per_h=10.0, v2_l=40.0)
    c = simulate(p, [Dose(0.0, 2.0, 600.0)], np.array([200.0]))
    assert c[0] == pytest.approx(600.0 / 60.0, rel=1e-6)  # equilibrated over V1 + V2


def test_bolus_initial_concentration():
    c = simulate(P, [Dose(0.0, 0.0, 400.0)], np.array([0.0]))
    assert c[0] == pytest.approx(400.0 / 20.0)


def test_doses_at_the_same_time_add():
    one = simulate(P, [Dose(0.0, 1.0, 100.0)], np.array([3.0]))
    two = simulate(P, [Dose(0.0, 1.0, 100.0), Dose(0.0, 1.0, 100.0)], np.array([3.0]))
    assert two[0] == pytest.approx(2 * one[0])
