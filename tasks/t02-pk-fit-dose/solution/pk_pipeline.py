#!/usr/bin/env python3
"""Reference solution for t02 (Python): fit, uncertainty, prediction, regimen.

Usage: python pk_pipeline.py --indir <dir with observations.csv, problem.json> --outdir <dir>

Methods:
  * Model solved exactly: on each interval with constant infusion rate the
    two-compartment ODE is linear with constant input, so the augmented matrix
    exponential gives the state at the next event time with no step error.
  * Fit: nonlinear least squares on log concentrations (the assay error is
    log-normal with known sigma, so this is maximum likelihood), in log-parameter
    space (keeps parameters positive and well scaled), trust-region reflective
    with bounds, multi-start to avoid local minima.
  * Uncertainty: Fisher information F = J^T J / sigma^2 with J = d log C / d log p.
  * Regimen: concentrations scale linearly with dose, so for each allowed tau the
    feasible doses form an interval; take the cheapest feasible daily dose.
"""

import argparse
import json
import os

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares


def kmat(p):
    cl, v1, q, v2 = p
    return np.array([[-(cl + q) / v1, q / v2], [q / v1, -q / v2]])


def step(k, x, dt, rate):
    if dt <= 0:
        return x
    m = np.zeros((3, 3))
    m[:2, :2] = k
    m[0, 2] = rate
    e = expm(m * dt)
    return e[:2, :2] @ x + e[:2, 2]


def conc(p, doses, t):
    k = kmat(p)
    events = sorted({0.0, *t.tolist(), *[d[0] for d in doses], *[d[0] + d[1] for d in doses]})
    rate = lambda a: sum(d[2] / d[1] for d in doses if d[0] <= a < d[0] + d[1])
    x = np.zeros(2)
    val = {0.0: 0.0}
    for a, b in zip(events[:-1], events[1:]):
        x = step(k, x, b - a, rate(0.5 * (a + b)))
        val[b] = x[0] / p[1]
    return np.array([val[ti] for ti in t.tolist()])


def steady_state(p, dose, tau, tinf):
    k = kmat(p)
    b = step(k, step(k, np.zeros(2), tinf, dose / tinf), tau - tinf, 0.0)
    x0 = np.linalg.solve(np.eye(2) - expm(k * tau), b)
    ts = np.linspace(0.0, tau, 2001)
    xe = step(k, x0, tinf, dose / tinf)
    c = [step(k, x0, t, dose / tinf)[0] if t <= tinf else step(k, xe, t - tinf, 0.0)[0] for t in ts]
    return max(max(c), xe[0]) / p[1], x0[0] / p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    obs = np.genfromtxt(os.path.join(a.indir, "observations.csv"), delimiter=",", names=True)
    t_obs, c_obs = obs["time_h"], obs["conc_mg_per_L"]
    prob = json.load(open(os.path.join(a.indir, "problem.json")))
    doses = [(d["start_h"], d["duration_h"], d["amount_mg"]) for d in prob["doses"]]
    sigma = prob["error_model"]["sigma_log"]

    def resid(lp):
        return np.log(np.maximum(conc(np.exp(lp), doses, t_obs), 1e-300)) - np.log(c_obs)

    rng = np.random.default_rng(12345)
    starts = [np.log([4.5, 20, 9, 32])] + [
        rng.uniform(np.log([2, 8, 2, 8]), np.log([12, 60, 40, 120])) for _ in range(12)
    ]
    bounds = (np.log([0.1, 1, 0.1, 1]), np.log([100, 500, 500, 2000]))
    best = min(
        (least_squares(resid, s, method="trf", bounds=bounds) for s in starts),
        key=lambda r: r.cost,
    )
    lp = best.x
    p = np.exp(lp)

    h = 1e-5
    J = np.column_stack(
        [(resid(lp + h * e) - resid(lp - h * e)) / (2 * h) for e in np.eye(4)]
    )
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(J.T @ J / sigma**2))))

    eig = np.sort(-np.linalg.eigvals(kmat(p)).real)[::-1]
    params = {
        "CL_L_per_h": p[0], "V1_L": p[1], "Q_L_per_h": p[2], "V2_L": p[3],
        "se_log": dict(zip(["CL", "V1", "Q", "V2"], se.tolist())),
        "t_half_alpha_h": float(np.log(2) / eig[0]),
        "t_half_beta_h": float(np.log(2) / eig[1]),
    }

    g = prob["prediction_grid_h"]
    grid = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 10)
    pred = conc(p, doses, grid)

    lo, hi = prob["design_targets_mg_per_L"]
    tinf = prob["maintenance_infusion_h"]
    best_reg = None
    for tau in prob["allowed_tau_h"]:
        pk1, tr1 = steady_state(p, 1.0, tau, tinf)
        d_min, d_max = lo / tr1, hi / pk1
        if d_min <= d_max and (best_reg is None or d_min * 24 / tau < best_reg[2]):
            best_reg = (tau, d_min, d_min * 24 / tau)
    if best_reg is None:
        raise SystemExit("no intermittent regimen reaches the targets")
    tau, dose, _ = best_reg
    peak, trough = steady_state(p, dose, tau, tinf)

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "params.json"), "w") as fh:
        json.dump(params, fh, indent=2)
    np.savetxt(os.path.join(a.outdir, "prediction.csv"), np.column_stack([grid, pred]),
               delimiter=",", header="time_h,conc_mg_per_L", comments="", fmt="%.8g")
    with open(os.path.join(a.outdir, "regimen.json"), "w") as fh:
        json.dump({"tau_h": tau, "dose_mg": dose, "infusion_h": tinf,
                   "predicted_peak_mg_per_L": peak, "predicted_trough_mg_per_L": trough},
                  fh, indent=2)


if __name__ == "__main__":
    main()
