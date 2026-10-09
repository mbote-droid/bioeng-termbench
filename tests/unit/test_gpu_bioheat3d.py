"""The GPU benchmark's 3D solver is verified before it is timed: it must
converge to the 1D spherical reference at second order, and every GPU
backend must reproduce the NumPy result (GPU tests skip without a device)."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

GPU_DIR = Path(__file__).resolve().parents[2] / "gpu"
spec = importlib.util.spec_from_file_location("bioheat3d", GPU_DIR / "bioheat3d.py")
bh3 = importlib.util.module_from_spec(spec)
sys.modules["bioheat3d"] = bh3
spec.loader.exec_module(bh3)


def _has_gpu() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def test_3d_solver_converges_to_the_1d_reference_at_second_order() -> None:
    errs = []
    for n in (32, 48):
        v = bh3.verify(bh3.Case(), n)
        errs.append(abs(v["peak_3d_C"] - v["peak_1d_C"]))
        assert abs(v["radius_3d_mm"] - v["radius_1d_mm"]) < 0.2
    order = math.log(errs[0] / errs[1]) / math.log(48 / 32)
    assert 1.8 < order < 2.3


def test_time_step_lands_on_the_power_switch_and_is_stable() -> None:
    case = bh3.Case()
    h = 2 * case.half_width_m / 64
    dt = bh3.stable_dt(h, case)
    assert case.heating_s / dt == pytest.approx(round(case.heating_s / dt))
    assert dt * 12 * bh3.K / (bh3.RHO_C * h * h) < 2.0


@pytest.mark.skipif(not _has_gpu(), reason="no CUDA device")
@pytest.mark.parametrize("backend", ["cupy", "cupy-fused"])
def test_gpu_backends_reproduce_numpy(backend) -> None:
    _, base, _ = bh3.run(bh3.Case(), 32, "numpy", max_steps=30)
    _, line, _ = bh3.run(bh3.Case(), 32, backend, max_steps=30)
    np.testing.assert_allclose(line, base, atol=1e-9)
