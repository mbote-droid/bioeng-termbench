"""Local test bench for your controller (a self-contained copy of the plant).

    python /app/data/simulate.py /app/controller.py [--patients 5] [--seed 0] [--octave]

Runs your controller in closed loop on example virtual patients, using the
same model, scenarios, protocol and metrics as the grader, and prints the
metrics per patient. The hidden patients are drawn from the same population
but with different random draws; passing here does not guarantee passing
there, so test many patients.
"""

import argparse
import json
import math
import select
import subprocess
import sys
import time

import numpy as np

DT, H, U_MAX = 5.0, 0.5, 120.0


def sample(rng):
    mass, p2, si = rng.uniform(55, 95), rng.uniform(0.02, 0.03), rng.uniform(3e-4, 8e-4)
    return dict(mass=mass, p1=rng.uniform(0.002, 0.008), p2=p2, p3=si * p2, n=rng.uniform(0.09, 0.15),
                gb=rng.uniform(105, 135), ub=0.0085 / si * (mass / 70) * rng.uniform(0.8, 1.25),
                tau=rng.uniform(5, 12))


def scenarios(rng):
    meals = [(t0 + rng.uniform(-30, 30), rng.uniform(30, 60), rng.uniform(40, 60)) for t0 in (60, 390, 780)]
    drop, spike, comp = rng.uniform(120, 1000), rng.uniform(1080, 1300), rng.uniform(1080, 1300)
    faults = [(drop, drop + 30, None), (spike, spike + 15, rng.uniform(60, 90)), (comp, comp + 20, -rng.uniform(40, 60))]
    ex = (DT * round(rng.uniform(570, 630) / DT), 45.0, rng.uniform(0.004, 0.008), rng.uniform(0.5, 1.0))
    return [("day", 1440.0, None, meals, faults, ex), ("correction", 360.0, 250.0, [], [], None)]


def deriv(p, x, t, u, meals, ex):
    g, xa, ins, gi = x
    ra = sum(1000 * c * 0.8 * (t - t0) / tm**2 * math.exp(-(t - t0) / tm) for t0, c, tm in meals if t > t0)
    k, a = 0.0, 1.0
    if ex and t >= ex[0]:
        s = t - ex[0]
        k, a = (ex[2], 1 + ex[3]) if s < ex[1] else (0.0, 1 + ex[3] * math.exp(-(s - ex[1]) / 90))
    vi, vg = 0.12 * p["mass"], 1.6 * p["mass"]
    ib = p["ub"] / (p["n"] * vi)
    return np.array([-p["p1"] * (g - p["gb"]) - (xa + (a - 1) * max(xa, 0)) * g - k * g + ra / vg,
                     -p["p2"] * xa + p["p3"] * (ins - ib), -p["n"] * ins + u / vi, (g - gi) / p["tau"]])


def run(cmd, p, sc, rng):
    name, dur, g0, meals, faults, ex = sc
    g0 = p["gb"] if g0 is None else g0
    x = np.array([g0, 0.0, p["ub"] / (p["n"] * 0.12 * p["mass"]), g0])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    drift, t, trace, buf = 0.0, 0.0, [g0], b""
    try:
        while t < dur - 1e-9:
            drift = 0.7 * drift + rng.normal(0, 3)
            cgm = x[3] + drift + rng.normal(0, 0.03 * x[3])
            for a, b, off in faults:
                if a <= t < b:
                    cgm = None if off is None or cgm is None else cgm + off
            cgm = None if cgm is None else round(float(np.clip(cgm, 40, 400)), 1)
            msg = {"t_min": t, "cgm_mg_dL": cgm, "basal_mU_per_min": p["ub"], "pump_max_mU_per_min": U_MAX,
                   "exercise_announced": bool(ex and ex[0] - 60 <= t < ex[0] + ex[1])}
            proc.stdin.write((json.dumps(msg) + "\n").encode())
            proc.stdin.flush()
            deadline = time.monotonic() + (30 if t == 0 else 2)
            while b"\n" not in buf:
                r, _, _ = select.select([proc.stdout], [], [], max(0, deadline - time.monotonic()))
                if not r:
                    return f"{name}: step deadline exceeded at t={t:g}"
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    return f"{name}: controller exited at t={t:g}"
                buf += chunk
            line, buf = buf.split(b"\n", 1)
            u = json.loads(line)["insulin_mU_per_min"]
            if not 0 <= u <= U_MAX:
                return f"{name}: requested {u} outside [0, {U_MAX}]"
            for k in range(int(DT / H)):
                tk = t + k * H
                k1 = deriv(p, x, tk, u, meals, ex)
                k2 = deriv(p, x + H / 2 * k1, tk + H / 2, u, meals, ex)
                k3 = deriv(p, x + H / 2 * k2, tk + H / 2, u, meals, ex)
                k4 = deriv(p, x + H * k3, tk + H, u, meals, ex)
                x = x + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                trace.append(x[0])
            t += DT
    finally:
        proc.kill()
    return np.array(trace)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controller")
    ap.add_argument("--patients", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--octave", action="store_true")
    a = ap.parse_args()
    cmd = ["octave-cli", "--no-gui", "--quiet", a.controller] if a.octave else [sys.executable, a.controller]
    rng = np.random.default_rng(a.seed)
    ok_all = True
    for i in range(a.patients):
        p = sample(rng)
        scs = scenarios(rng)
        res = [run(cmd, p, sc, rng) for sc in scs]
        if any(isinstance(r, str) for r in res):
            print(f"patient {i}: RUN FAILED: {next(r for r in res if isinstance(r, str))}")
            ok_all = False
            continue
        day, cor = res
        tir = np.mean((day[1:] >= 70) & (day[1:] <= 180))
        tar = np.mean(day[1:] > 250)
        low = min(day.min(), cor.min())
        below = np.flatnonzero(cor < 180)
        t180 = below[0] * H if below.size else math.inf
        late = cor[-240:]
        ok = (low >= 70 and tir >= 0.70 and tar <= 0.05 and t180 <= 120 and np.ptp(late) <= 50 and 80 <= late.mean() <= 160)
        ok_all &= ok
        print(f"patient {i}: {'PASS' if ok else 'FAIL'}  min {low:.0f}  TIR {tir:.1%}  >250 {tar:.1%}  "
              f"correction {t180:.0f} min  settle p2p {np.ptp(late):.0f}, mean {late.mean():.0f}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
