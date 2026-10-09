# GPU benchmark (outside grading)

The graded t04 solver is 1D: spherical symmetry makes it exact and cheap. Real
treatment planning (off-centre probes, vessels, anisotropic tissue) is 3D, and
3D is where a GPU pays. This folder holds a 3D Cartesian solver of the same
physics and a benchmark that times it on CPU and GPU.

| file | purpose |
|---|---|
| `bioheat3d.py` | 3D solver; one code path for NumPy and CuPy (`xp`), plus a fused CUDA kernel (`cupy-fused`) |
| `benchmark.py` | correctness gates, then throughput and end-to-end timing; writes `results/` |
| `BioEng_GPU_Benchmark.ipynb` | Colab notebook (free T4) that runs everything |
| `requirements.txt` | cupy + matplotlib (preinstalled on Colab) |

Correctness comes before speed:

1. The 3D solver converges to the 1D spherical reference at second order
   (measured order 2.02 between 33^3 and 49^3 nodes, CPU).
2. Each GPU backend must reproduce NumPy on the same grid. The fused kernel's
   logic was additionally compiled as C++ and run on the CPU; it matches the
   NumPy implementation bit for bit over 30 steps.

Run locally with a CUDA GPU: `pip install -r gpu/requirements.txt && python gpu/benchmark.py`.
Without a GPU the benchmark runs the NumPy parts only.

Results: `results/` is filled by running the notebook; the numbers are
measured, not estimated.
