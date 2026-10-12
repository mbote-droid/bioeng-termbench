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

## Results (Tesla T4, Colab)

Measured, in `results/` (`results.json`, `results.md`, `speedup.png`).

Correctness, on the GPU:

* CuPy and the fused kernel agree with NumPy to 7e-15 K (float64, same grid).
* The 3D solution converges to the 1D reference at observed orders 2.02,
  2.01, 2.00 (32^3 to 96^3).
* The full treatment at 257^3 nodes gives a peak of 96.08 C and an
  ablation radius of 9.206 mm. The 1D reference gives 96.06 C and
  9.204 mm.

Speed, in million cell updates per second:

| grid | NumPy f64 | CuPy f64 | fused f64 | NumPy f32 | CuPy f32 | fused f32 |
|---|---|---|---|---|---|---|
| 65^3 | 10 | 41 | 231 | 21 | 151 | 2807 |
| 129^3 | 9 | 161 | 607 | 22 | 414 | 4026 |
| 193^3 | 5 | 183 | 597 | 17 | 422 | 4004 |
| 257^3 | n/a | 188 | 581 | n/a | 426 | 3865 |

A full 4-minute treatment on the 257^3 grid (17 M nodes, 2233 steps)
takes 12 s with the fused kernel and 90 s with CuPy array code (float32).

What the numbers mean:

* **Kernel fusion is the main gain.** Array-style CuPy launches one kernel
  per operation and writes every temporary to memory. The fused kernel
  does a whole Heun stage in one pass and is 9x faster in float32.
* **float32 is bandwidth bound, float64 is not.** The fused kernel moves
  60 bytes per cell update in float32 (15 array reads and writes over two
  stages). At 3.9-4.0 G updates/s that is 230-240 GB/s, about 75 % of the
  T4's 320 GB/s. In float64 it moves 120 bytes at 0.6 G updates/s, about
  70 GB/s or 22 % of peak. The limit there is the T4's double-precision
  rate (1/32 of single): each node needs two exponentials and two
  divisions per stage. So float32 is 6.7x faster than float64 here, not
  the 2x a purely bandwidth-bound kernel would give. On a data-centre GPU
  with full float64 (A100, H100) the gap would close towards 2x.
* **Small grids do not fill the GPU.** At 65^3, launch overhead and low
  occupancy keep throughput below its plateau.
* **The NumPy baseline is plain array code on one CPU core**, so the
  110-230x speed-ups against it are not what an optimised multi-core CPU
  code would lose. The GPU-to-GPU comparisons (fused vs CuPy, float32 vs
  float64) do not depend on that baseline.
