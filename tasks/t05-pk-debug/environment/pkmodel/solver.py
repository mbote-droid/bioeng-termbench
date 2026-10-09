"""Numerical solution of the two-compartment model."""

import numpy as np
from scipy.integrate import solve_ivp


def _rate_matrix(p):
    k10, k12, k21 = p.cl_l_per_h / p.v1_l, p.q_l_per_h / p.v1_l, p.q_l_per_h / p.v2_l
    return np.array([[-(k10 + k12), k21], [k12, -k21]])


def simulate(p, doses, times):
    """Plasma concentration (mg/L) at the sorted `times` (h)."""
    k = _rate_matrix(p)

    def rate(t):
        total = 0.0
        for d in doses:
            if d.is_bolus:
                continue  # TODO: boluses
            if d.start_h <= t < d.start_h + d.duration_h:
                total += d.amount_mg / d.duration_h
        return total

    def rhs(t, x):
        dx = k @ x
        dx[0] += rate(t)
        return dx

    sol = solve_ivp(rhs, (0.0, float(times[-1])), np.zeros(2), t_eval=times, rtol=1e-3)
    return sol.y[0] / p.v1_l
