"""GPU benchmark: CPU (NumPy) vs GPU (CuPy arrays, CuPy fused kernel).

    python gpu/benchmark.py [--sizes 64 128 192 256] [--steps 40] [--out gpu/results]

1. Correctness first: every backend must reproduce the NumPy result on the
   same grid (same discretisation, so they agree to round-off in float64),
   and the 3D solution must converge to the 1D spherical reference at second
   order. A fast wrong answer is not a result.
2. Throughput: cell updates per second for a fixed number of steps, after a
   warm-up run (CUDA context, kernel compilation), synchronised timing.
3. End-to-end: wall time of a full treatment simulation at the largest size.

Writes results.json, results.md (table) and speedup.png.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bioheat3d import Case, run, verify


def gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def device_name() -> str:
    if not gpu_available():
        return "none"
    import cupy

    return cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()


def agreement(n: int, backends: list[str], dtype: str) -> dict[str, float]:
    """Max |difference| of the centre-line temperature after 30 steps, vs NumPy."""
    _, base, _ = run(Case(), n, "numpy", dtype, max_steps=30)
    out = {}
    for b in backends:
        _, line, _ = run(Case(), n, b, dtype, max_steps=30)
        out[b] = float(np.max(np.abs(line - base)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 192, 256])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--numpy-max", type=int, default=192, help="largest size timed on the CPU")
    ap.add_argument("--out", default="gpu/results")
    ap.add_argument("--skip-convergence", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = gpu_available()
    backends = ["numpy"] + (["cupy", "cupy-fused"] if gpu else [])
    report: dict = {
        "machine": {
            "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "gpu": device_name(),
        },
        "agreement_float64": {},
        "convergence": [],
        "throughput": [],
        "end_to_end": [],
    }

    # 1. correctness
    if gpu:
        report["agreement_float64"] = agreement(48, backends[1:], "float64")
        print("agreement vs numpy (float64, max |dT| in K):", report["agreement_float64"])
    if not a.skip_convergence:
        for n in (32, 48, 64, 96) if gpu else (32, 48):
            v = verify(Case(), n, "cupy-fused" if gpu else "numpy")
            report["convergence"].append(v)
            print(f"n={n}: peak 3D {v['peak_3d_C']:.3f} vs 1D {v['peak_1d_C']:.3f} C")
        errs = [abs(v["peak_3d_C"] - v["peak_1d_C"]) for v in report["convergence"]]
        ns = [v["n"] for v in report["convergence"]]
        report["observed_order"] = [
            float(np.log(errs[i] / errs[i + 1]) / np.log(ns[i + 1] / ns[i]))
            for i in range(len(ns) - 1)
        ]
        print("observed order of the 3D solver vs the 1D reference:", report["observed_order"])

    # 2. throughput
    for n in a.sizes:
        for b in backends:
            if b == "numpy" and n > a.numpy_max:
                continue
            for dtype in ("float64", "float32"):
                run(Case(), min(n, 64), b, dtype, max_steps=3)  # warm-up / compile
                t, _, _ = run(Case(), n, b, dtype, max_steps=a.steps)
                report["throughput"].append(asdict(t))
                print(
                    f"{b:11s} {dtype} n={n:4d}: {t.cell_updates_per_s / 1e6:9.1f} M cell-updates/s"
                )

    # 3. end to end at the largest GPU size
    if gpu:
        n = max(a.sizes)
        for b in ("cupy", "cupy-fused"):
            t, _, _ = run(Case(), n, b, "float32")
            report["end_to_end"].append(asdict(t))
            print(f"full treatment {b} float32 n={n}: {t.seconds:.1f} s for {t.steps} steps")

    (out / "results.json").write_text(json.dumps(report, indent=2))
    write_table(report, out / "results.md")
    plot(report, out / "speedup.png")
    print(f"wrote results to {out}/")


def write_table(report: dict, path: Path) -> None:
    rows = report["throughput"]
    base = {(r["n"], r["dtype"]): r["cell_updates_per_s"] for r in rows if r["backend"] == "numpy"}
    lines = [
        f"GPU: {report['machine']['gpu']}  CPU: {report['machine']['cpu']}",
        "",
        "| backend | dtype | n | cells | M cell-updates/s | speed-up vs NumPy |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ref = base.get((r["n"], r["dtype"]))
        speed = f"{r['cell_updates_per_s'] / ref:.1f}x" if ref else "n/a (CPU too slow)"
        lines.append(
            f"| {r['backend']} | {r['dtype']} | {r['n']} | {r['cells']:,} | "
            f"{r['cell_updates_per_s'] / 1e6:.1f} | {speed} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(report: dict, path: Path) -> None:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed: skipping the plot")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (b, dtype), style in {
        ("numpy", "float64"): "o-",
        ("cupy", "float64"): "s-",
        ("cupy-fused", "float64"): "^-",
        ("numpy", "float32"): "o--",
        ("cupy", "float32"): "s--",
        ("cupy-fused", "float32"): "^--",
    }.items():
        pts = [
            (r["cells"], r["cell_updates_per_s"])
            for r in report["throughput"]
            if r["backend"] == b and r["dtype"] == dtype
        ]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.loglog(xs, ys, style, label=f"{b} {dtype}")
    ax.set_xlabel("grid nodes")
    ax.set_ylabel("cell updates per second")
    ax.set_title(f"3D bioheat (Heun, 7-point stencil): {report['machine']['gpu']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
