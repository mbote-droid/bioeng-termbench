"""3D Cartesian bioheat solver for the GPU benchmark (outside grading).

Same physics as t04 (Pennes with logistic perfusion collapse, Gaussian
source, Arrhenius damage) on a uniform (n+1)^3 vertex grid over the cube
[-L, L]^3, T = T_body on the faces. Second-order central differences in
space (7-point Laplacian) and Heun's method (explicit RK2) in time, with the
step at 90 % of the stability limit.

One code path serves every backend through the array namespace `xp`
(numpy or cupy): the physics is written once, the hardware is a parameter.
A third backend, `cupy-fused`, replaces the per-step array expressions by
one hand-written CUDA kernel (RawKernel) per Heun stage, to show what kernel
fusion buys over array-at-a-time GPU code (memory traffic, not arithmetic,
bounds a stencil).

Why 3D for the benchmark: the graded t04 solver is 1D (spherical symmetry
makes it exact and cheap); a real treatment plan has vessels, an off-centre
probe and anisotropy, so it is 3D, and 3D is where the GPU matters. The
spherically symmetric case is kept so that the 3D result can be checked
against the converged 1D reference (`verify`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

T_BODY = 37.0
K, RHO_C, RHOB_CB = 0.52, 1060.0 * 3600.0, 1060.0 * 3640.0
LN_A, EA_R = math.log(3.1e98), 6.28e5 / 8.314
STASIS_C, STASIS_W = 60.0, 3.0


@dataclass(frozen=True)
class Case:
    half_width_m: float = 0.04
    sigma_m: float = 0.007
    power_w: float = 6.0
    perfusion: float = 0.004
    heating_s: float = 180.0
    end_s: float = 240.0


@dataclass
class Timing:
    backend: str
    dtype: str
    n: int
    cells: int
    steps: int
    seconds: float
    cell_updates_per_s: float
    peak_c: float
    radius_mm: float


def get_xp(backend: str):
    if backend == "numpy":
        return np
    import cupy

    return cupy


def stable_dt(h: float, case: Case) -> float:
    """Heun on the diffusion operator: stable for dt <= 2 / lambda_max with
    lambda_max = 12 alpha / h^2 + perfusion term; use 90 % of it."""
    lam = 12.0 * K / (RHO_C * h * h) + case.perfusion * RHOB_CB / RHO_C
    dt = 0.9 * 2.0 / lam
    n_heat = math.ceil(case.heating_s / dt)
    return case.heating_s / n_heat  # land exactly on the power switch


def _laplacian(xp, th, h2):
    lap = xp.zeros_like(th)
    c = th[1:-1, 1:-1, 1:-1]
    lap[1:-1, 1:-1, 1:-1] = (
        th[2:, 1:-1, 1:-1]
        + th[:-2, 1:-1, 1:-1]
        + th[1:-1, 2:, 1:-1]
        + th[1:-1, :-2, 1:-1]
        + th[1:-1, 1:-1, 2:]
        + th[1:-1, 1:-1, :-2]
        - 6.0 * c
    ) / h2
    return lap


def _rhs(xp, th, q, w0, h2):
    """d(theta)/dt and d(Omega)/dt; theta = T - T_body, zero on the faces."""
    t = T_BODY + th
    collapse = 1.0 / (1.0 + xp.exp((t - STASIS_C) / STASIS_W))
    dth = (K * _laplacian(xp, th, h2) - w0 * collapse * th + q) / RHO_C
    dth[0, :, :] = dth[-1, :, :] = dth[:, 0, :] = dth[:, -1, :] = dth[:, :, 0] = dth[:, :, -1] = 0.0
    dom = xp.exp(xp.minimum(LN_A - EA_R / (t + 273.15), 80.0))
    return dth, dom


FUSED_SRC = r"""
extern "C" __global__
void heun_stage(const real_t* th, const real_t* om, const real_t* q, const real_t* th_base, const real_t* k1th, const real_t* k1om,
                real_t* out_th, real_t* out_om, real_t* k_th, real_t* k_om,
                const int n1, const real_t h2, const real_t dt, const real_t qscale, const int stage,
                const real_t w0, const real_t k, const real_t rho_c, const real_t tb,
                const real_t ln_a, const real_t ea_r, const real_t ts, const real_t tw) {
    // One thread per grid node. stage 0: k1 = f(th), out = th + dt k1.
    // stage 1: k2 = f(th_pred), out = th_base + dt/2 (k1 + k2).
    long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
    long total = (long)n1 * n1 * n1;
    if (idx >= total) return;
    int i = idx / (n1 * n1), j = (idx / n1) % n1, l = idx % n1;
    real_t x = th[idx];
    real_t t = tb + x;
    real_t rate = exp(fmin(ln_a - ea_r / (t + (real_t)273.15), (real_t)80));
    real_t dth = 0;
    if (i > 0 && j > 0 && l > 0 && i < n1 - 1 && j < n1 - 1 && l < n1 - 1) {
        long s = (long)n1 * n1;
        real_t lap = (th[idx + s] + th[idx - s] + th[idx + n1] + th[idx - n1] + th[idx + 1] + th[idx - 1] - 6 * x) / h2;
        real_t collapse = 1 / (1 + exp((t - ts) / tw));
        dth = (k * lap - w0 * collapse * x + qscale * q[idx]) / rho_c;
    }
    if (stage == 0) {
        k_th[idx] = dth; k_om[idx] = rate;
        out_th[idx] = x + dt * dth; out_om[idx] = om[idx] + dt * rate;
    } else {
        out_th[idx] = th_base[idx] + (real_t)0.5 * dt * (k1th[idx] + dth);
        out_om[idx] = om[idx] + (real_t)0.5 * dt * (k1om[idx] + rate);
    }
}
"""


def run(
    case: Case, n: int, backend: str = "numpy", dtype: str = "float64", max_steps: int | None = None
):
    """Simulate; returns (Timing, centre-line theta and Omega as numpy arrays).
    `max_steps` truncates the run for throughput-only timing."""
    fused = backend == "cupy-fused"
    xp = get_xp("cupy" if fused else backend)
    dt_type = xp.float64 if dtype == "float64" else xp.float32
    h = 2 * case.half_width_m / n
    x = xp.linspace(-case.half_width_m, case.half_width_m, n + 1, dtype=dt_type)
    r2 = x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2
    q = (case.power_w / (math.pi**1.5 * case.sigma_m**3) * xp.exp(-r2 / case.sigma_m**2)).astype(
        dt_type
    )
    del r2
    th = xp.zeros((n + 1,) * 3, dtype=dt_type)
    om = xp.zeros_like(th)
    w0 = case.perfusion * RHOB_CB
    dt = stable_dt(h, case)
    n_heat, n_total = round(case.heating_s / dt), round(case.end_s / dt)
    steps = n_total if max_steps is None else min(n_total, max_steps)
    c = n // 2
    peak = 0.0
    if fused:
        real_t = "double" if dtype == "float64" else "float"
        kernel = xp.RawKernel(f"typedef {real_t} real_t;\n" + FUSED_SRC, "heun_stage")
        buf = [xp.empty_like(th) for _ in range(6)]
        threads = 256
        blocks = ((n + 1) ** 3 + threads - 1) // threads
        sc = np.float64 if dtype == "float64" else np.float32
        consts = [sc(v) for v in (w0, K, RHO_C, T_BODY, LN_A, EA_R, STASIS_C, STASIS_W)]
    sync = xp.cuda.Stream.null.synchronize if xp is not np else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for step in range(steps):
        qs = 1.0 if step < n_heat else 0.0
        if fused:
            th_p, om_p, k1t, k1o, th_n, om_n = buf
            args1 = (
                th,
                om,
                q,
                th,
                th,
                om,
                th_p,
                om_p,
                k1t,
                k1o,
                np.int32(n + 1),
                sc(h * h),
                sc(dt),
                sc(qs),
                np.int32(0),
                *consts,
            )
            kernel((blocks,), (threads,), args1)
            args2 = (
                th_p,
                om,
                q,
                th,
                k1t,
                k1o,
                th_n,
                om_n,
                k1t,
                k1o,
                np.int32(n + 1),
                sc(h * h),
                sc(dt),
                sc(qs),
                np.int32(1),
                *consts,
            )
            kernel((blocks,), (threads,), args2)
            buf = [th_p, om_p, k1t, k1o, th, om]
            th, om = th_n, om_n
        else:
            k1, r1 = _rhs(xp, th, qs * q, w0, h * h)
            k2, r2_ = _rhs(xp, th + dt * k1, qs * q, w0, h * h)
            th = th + 0.5 * dt * (k1 + k2)
            om = om + 0.5 * dt * (r1 + r2_)
        if (step < n_heat + 5 and step >= n_heat - 5) or step % 50 == 0:
            peak = max(peak, float(th[c, c, c]))
    sync()
    seconds = time.perf_counter() - t0
    line_th = np.asarray(th[c:, c, c].get() if xp is not np else th[c:, c, c], dtype=np.float64)
    line_om = np.asarray(om[c:, c, c].get() if xp is not np else om[c:, c, c], dtype=np.float64)
    radius = ablation_radius_mm(line_om, h)
    cells = (n + 1) ** 3
    timing = Timing(
        backend, dtype, n, cells, steps, seconds, cells * steps / seconds, T_BODY + peak, radius
    )
    return timing, line_th, line_om


def ablation_radius_mm(omega_line: np.ndarray, h: float) -> float:
    inside = np.flatnonzero(omega_line >= 1.0)
    if inside.size == 0:
        return 0.0
    i = int(inside[-1])
    if i + 1 >= omega_line.size:
        return i * h * 1e3
    a, b = math.log(omega_line[i]), math.log(max(omega_line[i + 1], 1e-300))
    return (i - a / (b - a)) * h * 1e3


def verify(case: Case, n: int, backend: str = "numpy", dtype: str = "float64") -> dict:
    """Compare the 3D run with the converged 1D spherical reference."""
    from bioengbench.models.bioheat import Problem, simulate

    timing, _, _ = run(case, n, backend, dtype)
    ref = simulate(
        Problem(sigma_m=case.sigma_m, domain_radius_m=case.half_width_m),
        power_w=case.power_w,
        heating_s=case.heating_s,
        perfusion_per_s=case.perfusion,
        end_s=case.end_s,
        nr=800,
        dt=0.25,
    )
    return {
        "n": n,
        "backend": backend,
        "dtype": dtype,
        "peak_3d_C": timing.peak_c,
        "peak_1d_C": ref.t_center_peak_c,
        "radius_3d_mm": timing.radius_mm,
        "radius_1d_mm": ref.ablation_radius_mm,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=("numpy", "cupy", "cupy-fused"), default="numpy")
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        print(json.dumps(verify(Case(), a.n, a.backend, a.dtype), indent=2))
    else:
        timing, _, _ = run(Case(), a.n, a.backend, a.dtype, a.max_steps)
        print(json.dumps(asdict(timing), indent=2))


if __name__ == "__main__":
    main()
