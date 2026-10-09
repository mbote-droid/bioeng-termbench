"""Exact solution of the linear two-compartment model.

Between consecutive event times (dose starts and ends, output times) the
infusion rate is constant, so the system is linear time-invariant and is
advanced exactly with the matrix exponential of the augmented system
[[K, r], [0, 0]]. No step size, no tolerance, no stiffness problem: the
distribution phase can be arbitrarily fast. Boluses are jumps in A1.
"""

import itertools

import numpy as np
from scipy.linalg import expm


def _rate_matrix(p):
    k10, k12, k21 = p.cl_l_per_h / p.v1_l, p.q_l_per_h / p.v1_l, p.q_l_per_h / p.v2_l
    return np.array([[-(k10 + k12), k21], [k12, -k21]])


def _advance(k, x, dt, rate):
    if dt <= 0.0:
        return x
    m = np.zeros((3, 3))
    m[:2, :2] = k
    m[0, 2] = rate
    e = expm(m * dt)
    return e[:2, :2] @ x + e[:2, 2]


def simulate(p, doses, times):
    """Plasma concentration (mg/L) at the sorted `times` (h)."""
    k = _rate_matrix(p)
    infusions = [d for d in doses if not d.is_bolus]
    events = {0.0, *map(float, times)}
    for d in infusions:
        events |= {d.start_h, d.start_h + d.duration_h}
    for d in doses:
        events.add(d.start_h)
    grid = sorted(e for e in events if e >= 0.0)

    def rate(t):
        return sum(d.amount_mg / d.duration_h for d in infusions if d.start_h <= t < d.start_h + d.duration_h)

    def boluses_at(t):
        return sum(d.amount_mg for d in doses if d.is_bolus and d.start_h == t)

    x = np.zeros(2)
    x[0] += boluses_at(0.0)
    conc = {0.0: x[0] / p.v1_l}
    for a, b in itertools.pairwise(grid):
        x = _advance(k, x, b - a, rate(0.5 * (a + b)))
        x[0] += boluses_at(b)
        conc[b] = x[0] / p.v1_l
    return np.array([conc[float(t)] for t in times])
