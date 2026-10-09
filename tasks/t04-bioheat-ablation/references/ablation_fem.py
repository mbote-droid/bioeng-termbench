"""Independent t04 reference: vertex-centred P1 finite elements.

A different discretisation from the finite-volume solution, used to show that
the grader's tolerances accept any correct second-order method, not just the
grader's own scheme. Nodes r_j = j h (j = 0..N), Dirichlet at r = R; weak form
with the spherical weight r^2; lumped mass and lumped (nodal) perfusion;
consistent load vector by 3-point Gauss quadrature; Crank-Nicolson with
Picard iteration. Same command line as solution/ablation.py.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import expit

GAUSS_X = np.array([-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)])
GAUSS_W = np.array([5 / 9, 8 / 9, 5 / 9])


def simulate(cfg, power, heating, perfusion, nr, dt):
    tis, arr = cfg["tissue"], cfg["arrhenius"]
    k, rho_c = tis["conductivity_W_per_m_K"], tis["density_kg_per_m3"] * tis["specific_heat_J_per_kg_K"]
    tb, big_r, sg = cfg["body_temperature_C"], cfg["domain_radius_m"], cfg["source_sigma_m"]
    n_on, n_steps = round(heating / dt), round((heating + cfg["cooling_time_s"]) / dt)
    h = big_r / nr
    r = np.arange(nr + 1) * h
    # Element integrals with weight 4 pi r^2 (exact for the polynomial terms).
    a, b = r[:-1], r[1:]
    stiff = 4 * math.pi * k * (b**3 - a**3) / (3 * h * h)  # per element, times [[1,-1],[-1,1]]
    lump = np.zeros(nr + 1)  # lumped mass: integral of phi_i * 4 pi r^2
    lump[:-1] += 4 * math.pi * (b**4 / 4 - a * b**3 / 3 + a**4 / 12) / h  # exact, left node
    lump[1:] += 4 * math.pi * (b**4 / 12 - a**3 * b / 3 + a**4 / 4) / h  # exact, right node
    load = np.zeros(nr + 1)
    for xg, wg in zip(GAUSS_X, GAUSS_W):
        rq = 0.5 * (a + b) + 0.5 * h * xg
        q = power / (math.pi**1.5 * sg**3) * np.exp(-(rq / sg) ** 2) * 4 * math.pi * rq**2 * 0.5 * h * wg
        load[:-1] += q * (b - rq) / h
        load[1:] += q * (rq - a) / h
    n = nr  # unknowns: nodes 0..N-1 (node N fixed at theta = 0)
    diag = np.zeros(nr + 1)
    diag[:-1] += stiff
    diag[1:] += stiff
    off = -stiff  # couples j and j+1
    cap = rho_c * lump[:n]
    wfull = perfusion * tis["blood_volumetric_heat_capacity_J_per_m3_K"] * lump[:n]

    def collapse(th):
        return expit((tis["stasis_temperature_C"] - tb - th) / tis["stasis_width_K"])

    def rate(th):
        return arr["A_per_s"] * np.exp(-arr["Ea_J_per_mol"] / (arr["R_J_per_mol_K"] * (tb + th + 273.15)))

    def kmul(th):
        out = diag[:n] * th
        out[:-1] += off[: n - 1] * th[1:]
        out[1:] += off[: n - 1] * th[:-1]
        return out

    th, om = np.zeros(n), np.zeros(n)
    ab = np.zeros((3, n))
    ab[0, 1:] = 0.5 * dt * off[: n - 1]
    ab[2, :-1] = 0.5 * dt * off[: n - 1]
    peak = e_on = dep = perf = 0.0
    bnd = 0.0
    for step in range(n_steps):
        f = load[:n] if step < n_on else 0.0 * load[:n]
        w_old = wfull * collapse(th)
        rhs = cap * th - 0.5 * dt * (kmul(th) + w_old * th) + dt * f
        th0, om0, w = th, om, w_old
        for _ in range(3):
            w_used = w
            ab[1] = cap + 0.5 * dt * (diag[:n] + w_used)
            th = solve_banded((1, 1), ab, rhs)
            om = om0 + 0.5 * dt * (rate(th0) + rate(th))
            w = wfull * collapse(th)
        dep += dt * f.sum()
        perf += 0.5 * dt * (w_old @ th0 + w_used @ th)
        # Boundary flux = reaction at the fixed node: -(K theta)_N, from the last element.
        bnd += 0.5 * dt * (-off[n - 1]) * (th0[-1] + th[-1])
        peak = max(peak, th[0])
        if step + 1 == n_on:
            e_on = cap @ th
    inside = np.flatnonzero(om >= 1)
    if inside.size == 0:
        radius = 0.0
    else:
        i = inside[-1]
        if i + 1 >= n:
            radius = r[n - 1] * 1e3
        else:
            la, lb = math.log(om[i]), math.log(max(om[i + 1], 1e-300))
            radius = (r[i] - la / (lb - la) * h) * 1e3
    return {
        "t_center_peak_C": tb + peak,
        "energy_at_heating_end_J": e_on,
        "ablation_radius_mm": radius,
        "energy_budget_J": {"deposited": dep, "stored": cap @ th, "perfusion": perf, "boundary": bnd},
    }


def plan(cfg):
    lo, hi = cfg["tissue"]["perfusion_range_per_s"]
    need = cfg["tumour_radius_mm"] + cfg["margin_mm"]
    best = None
    for t in cfg["heating_times_s"]:
        p_hi = brentq(lambda p, t=t: simulate(cfg, p, t, lo, 200, 2.0)["t_center_peak_C"] - cfg["peak_limit_C"], 0.1, 200.0, xtol=1e-4)
        if simulate(cfg, p_hi, t, hi, 200, 2.0)["ablation_radius_mm"] < need:
            continue
        p_lo = brentq(lambda p, t=t: simulate(cfg, p, t, hi, 200, 2.0)["ablation_radius_mm"] - need, 0.1, p_hi, xtol=1e-4)
        p = min(1.01 * p_lo, 0.5 * (p_lo + p_hi))
        if best is None or p * t < best[0] * best[1]:
            best = (p, t)
    p, t = best
    return {
        "heating_time_s": t,
        "power_W": p,
        "predicted_ablation_radius_mm": simulate(cfg, p, t, hi, 400, 1.0)["ablation_radius_mm"],
        "predicted_peak_C": simulate(cfg, p, t, lo, 400, 1.0)["t_center_peak_C"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("simulate", "plan"))
    ap.add_argument("--problem", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--power", type=float)
    ap.add_argument("--heating-time", type=float)
    ap.add_argument("--perfusion", type=float)
    ap.add_argument("--nr", type=int, default=400)
    ap.add_argument("--dt", type=float, default=1.0)
    a = ap.parse_args()
    cfg = json.loads(Path(a.problem).read_text())
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "simulate":
        res = simulate(cfg, a.power, a.heating_time, a.perfusion, a.nr, a.dt)
        (out / "simulation.json").write_text(json.dumps(res, indent=2))
    else:
        (out / "plan.json").write_text(json.dumps(plan(cfg), indent=2))


if __name__ == "__main__":
    main()
