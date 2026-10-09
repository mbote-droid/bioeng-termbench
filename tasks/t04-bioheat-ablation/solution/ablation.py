"""Reference solution for t04: bioheat solver + robust ablation planner.

    python ablation.py simulate --problem P.json --power W --heating-time S \
        --perfusion OMEGA --nr N --dt DT --outdir DIR
    python ablation.py plan --problem P.json --outdir DIR

Physics (spherical symmetry, theta = T - T_body):

    rho c dtheta/dt = (1/r^2) d/dr(k r^2 dtheta/dr) - omega rho_b c_b s(T) theta + Q
    s(T) = 1 / (1 + exp((T - T_s) / dT_s))   (vascular collapse)
    dOmega/dt = A exp(-Ea / (R (T + 273.15)))

Numerics: cell-centred finite volumes (second order in space), Crank-Nicolson
in time with Picard iteration for the temperature-dependent perfusion (lagging the
perfusion would make the scheme first order), trapezoidal damage integral,
exact shell integrals of the Gaussian source, and an energy budget computed
with the same quadratures so it closes to round-off.

Planning: for each allowed heating time, the feasible powers form an interval
[P_lo, P_hi]: the lesion radius grows with power, so coverage at the highest
perfusion sets P_lo; the centre peak grows with power, so the 100 C limit at
the lowest perfusion sets P_hi. Energy P*t is smallest at the lowest feasible
power; pick the heating time with the smallest P_lo*t, then aim 1 % above
P_lo (covers solver and root-finding error) while staying below P_hi.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import erf, expit

SEARCH = (200, 2.0)
POWER_MARGIN = 1.01


def simulate(cfg, power, heating, perfusion, nr, dt):
    tis, arr = cfg["tissue"], cfg["arrhenius"]
    k, rho_c = tis["conductivity_W_per_m_K"], tis["density_kg_per_m3"] * tis["specific_heat_J_per_kg_K"]
    rbcb, t_body = tis["blood_volumetric_heat_capacity_J_per_m3_K"], cfg["body_temperature_C"]
    big_r, sigma = cfg["domain_radius_m"], cfg["source_sigma_m"]
    end = heating + cfg["cooling_time_s"]
    n_on, n_steps = round(heating / dt), round(end / dt)

    h = big_r / nr
    f = np.arange(nr + 1) * h
    rc = 0.5 * (f[:-1] + f[1:])
    vol = 4 * np.pi / 3 * (f[1:] ** 3 - f[:-1] ** 3)
    cap = rho_c * vol
    g = k * 4 * np.pi * f[1:-1] ** 2 / h
    g_b = k * 4 * np.pi * big_r**2 / (h / 2)
    x = f / sigma
    cum = erf(x) - 2 * x / np.sqrt(np.pi) * np.exp(-(x**2))
    src = power * np.diff(cum)
    w_full = perfusion * rbcb * vol
    diag = np.zeros(nr)
    diag[:-1] -= g
    diag[1:] -= g
    diag[-1] -= g_b

    stasis_c, stasis_w = tis["stasis_temperature_C"], tis["stasis_width_K"]

    def collapse(th):
        return expit((stasis_c - t_body - th) / stasis_w)

    def rate(th):
        return arr["A_per_s"] * np.exp(-arr["Ea_J_per_mol"] / (arr["R_J_per_mol_K"] * (t_body + th + 273.15)))

    th, om = np.zeros(nr), np.zeros(nr)
    ab = np.zeros((3, nr))
    ab[0, 1:] = ab[2, :-1] = -0.5 * dt * g
    peak = e_on = dep = perf = bnd = 0.0
    for n in range(n_steps):
        q = src if n < n_on else 0.0 * src
        w_old = w_full * collapse(th)
        lth = diag * th
        lth[:-1] += g * th[1:]
        lth[1:] += g * th[:-1]
        rhs = cap * th + 0.5 * dt * (lth - w_old * th) + dt * q
        th0, om0, w = th, om, w_old
        for _ in range(3):
            w_used = w
            ab[1] = cap - 0.5 * dt * diag + 0.5 * dt * w_used
            th = solve_banded((1, 1), ab, rhs)
            om = om0 + 0.5 * dt * (rate(th0) + rate(th))
            w = w_full * collapse(th)
        dep += dt * q.sum()
        perf += 0.5 * dt * (w_old @ th0 + w_used @ th)
        bnd += 0.5 * dt * g_b * (th0[-1] + th[-1])
        peak = max(peak, (9 * th[0] - th[1]) / 8)
        if n + 1 == n_on:
            e_on = cap @ th
    inside = np.flatnonzero(om >= 1)
    if inside.size == 0:
        radius = 0.0
    elif inside[-1] + 1 >= nr:
        radius = rc[-1] * 1e3
    else:
        i = inside[-1]
        a, b = math.log(om[i]), math.log(max(om[i + 1], 1e-300))
        radius = (rc[i] - a / (b - a) * h) * 1e3
    return {
        "t_center_peak_C": t_body + peak,
        "energy_at_heating_end_J": e_on,
        "ablation_radius_mm": radius,
        "energy_budget_J": {"deposited": dep, "stored": cap @ th, "perfusion": perf, "boundary": bnd},
    }


def plan(cfg):
    lo, hi = cfg["tissue"]["perfusion_range_per_s"]
    need = cfg["tumour_radius_mm"] + cfg["margin_mm"]
    limit = cfg["peak_limit_C"]
    best = None
    for t in cfg["heating_times_s"]:
        def peak(p, t=t):
            return simulate(cfg, p, t, lo, *SEARCH)["t_center_peak_C"] - limit

        def cover(p, t=t):
            return simulate(cfg, p, t, hi, *SEARCH)["ablation_radius_mm"] - need

        p_hi = brentq(peak, 0.1, 200.0, xtol=1e-4)
        if cover(p_hi) < 0:
            continue  # no power reaches the margin without exceeding the limit
        p_lo = brentq(cover, 0.1, p_hi, xtol=1e-4)
        p = min(p_lo * POWER_MARGIN, 0.5 * (p_lo + p_hi))
        if best is None or p * t < best[0] * best[1]:
            best = (p, t)
    if best is None:
        raise SystemExit("no feasible plan")
    p, t = best
    at_hi = simulate(cfg, p, t, hi, 400, 1.0)
    at_lo = simulate(cfg, p, t, lo, 400, 1.0)
    return {
        "heating_time_s": t,
        "power_W": p,
        "predicted_ablation_radius_mm": at_hi["ablation_radius_mm"],
        "predicted_peak_C": at_lo["t_center_peak_C"],
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
