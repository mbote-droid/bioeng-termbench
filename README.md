# BioEng-TermBench
[![CI](https://github.com/mbote-droid/bioeng-termbench/actions/workflows/ci.yml/badge.svg)](https://github.com/mbote-droid/bioeng-termbench/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/deps-uv-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


Terminal-based **biomedical engineering tasks with physics-aware graders**, built
to train and evaluate AI agents. Each task turns a real engineering workflow into
a reproducible challenge in the [Harbor](https://harborframework.com) task format,
and each grader checks engineering correctness (units, conservation, order of
accuracy, safety at the true parameters, robustness under uncertainty), not just
"the script ran".

**Why it exists.** An agent trained on a reward learns whatever the grader
accepts. A grader that only checks that output files exist will reward unit
slips, unconverged solvers and controllers that are fine on average but
dangerous for some patients. Here the grader is treated as the *sensor* in the
training loop and is itself validated like an instrument: correct solutions
written with different methods and languages must pass, realistic wrong
solutions must fail with the right diagnostic code, every check must be
necessary, and the sandbox must hold against attacks.

## Tasks

| task | workflow | the decision the maths drives | references |
|---|---|---|---|
| `t01-ecg-denoise-rpeaks` | denoise an ambulatory ECG, detect beats, HRV | filter design: suppression vs morphology fidelity | Python x2, Octave |
| `t02-pk-fit-dose` | fit a two-compartment PK model to 14 noisy samples | a dosing regimen that is safe for the *true* patient | Python, Octave (Julia port, unverified) |
| `t03-glucose-closed-loop` | write an insulin pump controller, tested in the loop | insulin every 5 min through meals, exercise and sensor faults | Python, Octave |
| `t04-bioheat-ablation` | verified bioheat solver + thermal ablation plan | heating time and power, robust to +-20 % perfusion | finite volumes, finite elements, Octave |
| `t05-pk-debug` | find and fix five defects in a PK package | what is wrong, judged against a specification | fixed package |

Details: [docs/DESIGN.md](docs/DESIGN.md). Equations and numerical methods:
[docs/EQUATIONS.md](docs/EQUATIONS.md).

## Quick start

    uv sync --frozen        # locked environment (Python 3.12)
    make check              # lint, strict typing, full test suite
    uv run bioeng list
    uv run bioeng grade t03-glucose-closed-loop \
        --submission tasks/t03-glucose-closed-loop/solution/controller.py \
        --seeds tasks/t03-glucose-closed-loop/tests/dev_seeds.txt

As root the grader refuses to run a submission without `--sandbox-user`
(see [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)). Every terminal workflow,
from daily checks to the validation campaign and image builds:
[docs/WORKFLOWS.md](docs/WORKFLOWS.md).

## Grader validation

`make validate` runs the campaign as a Snakemake workflow and regenerates
[docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md), where every number is
linked to a result file with provenance and a SHA-256 manifest:

* **references**: independent correct solutions on many hidden cases; every
  failure is a false reject (Wilson / rule-of-three bounds);
* **negative controls**: realistic mistakes, mostly one-edit mutants of a
  reference; each must fail with its expected code, and its chance of
  earning reward over 8 hidden cases must be below 5 %;
* **grader mutation testing**: switch a check off, or move a threshold, and
  the test suite must notice;
* **red team**: seven attacks on the sandbox (escape, symlinks, fork bomb,
  detached daemon, disk, memory, forging t03's simulation record).

Findings the process produced, not anticipated (full list with evidence in
[validation/FINDINGS.md](validation/FINDINGS.md)):

* t02's Wald test rejected correct fits (weak identifiability); replaced by
  a likelihood-ratio region.
* A plain PID passed 90 % of t03 patients until exercise was modelled.
* t04: damage-driven perfusion made second order unmeasurable; joint
  space-time refinement let error terms cancel; and the centre temperature
  is a poor convergence functional (the independent FEM reference failed on
  it in 2 of 8 cases), so the order is measured on the stored energy.
* A t01 reference had an edge-detection bug (1 recording in 100); the
  grader was right.
* The t01 median-baseline method sits on the 0.05 mV fidelity limit: it
  passes about half the recordings and fails the rest on FIDELITY only.
* Two planted "mistakes" were acceptable solutions (equivalent mutants) and
  are kept as references.
* Without the sandbox user a t03 controller forges its own simulation
  record and earns a pass.
* Mutation testing showed most acceptance thresholds were unpinned; boundary
  tests now pin each one with the contract's numbers.

## GPU benchmark

`gpu/` holds a 3D version of the t04 solver (one code path for NumPy and
CuPy, plus a fused CUDA kernel) and a benchmark that checks correctness
before it times anything: the 3D solution converges to the 1D reference at
measured order 2.0, and the fused kernel was compiled as C++ and matched the
NumPy implementation bit for bit. Run it on a free Colab T4 with
[gpu/BioEng_GPU_Benchmark.ipynb](gpu/BioEng_GPU_Benchmark.ipynb).

## Evaluating real agents

`agent_eval/harness.py` runs a tool-use agent loop against the Anthropic API
(your key, from the environment, never written to disk), grades the result
with the real, sandboxed grader and keeps the full transcript. Harbor is the
faithful path when Docker is available
(`harbor run -p tasks/<id> -a <agent> -m <model>`).

## Layout

    src/bioengbench/
      core/      lifecycle (Template Method), registry, sandboxed runner
                 (local or Docker), interactive sessions, seeds, scoring
      checks/    strict readers, signal metrics, event matching, units, statistics
      models/    ECG (McSharry), two-compartment PK, Bergman glucose-insulin,
                 Pennes bioheat with Arrhenius damage
      tasks/     one plugin per task: case generator + check suite
      tooling/   the `bioeng` CLI
    tasks/<id>/  Harbor task: instruction.md, task.toml, environment/,
                 solution/, tests/ (verifier), references/ (validation only)
    validation/  probes, campaign, grader mutation, red team, report
    workflow/    Snakefile for the campaign
    gpu/         3D solver, CPU/GPU benchmark, Colab notebook
    agent_eval/  agent harness
    docker/      agent and grader base images, hash-pinned requirements
    docs/        requirements, architecture (ADRs), design, ICD, equations,
                 diagrams, threat model, FMEA, data, workflows, validation report

## Documentation

| document | contents |
|---|---|
| [REQUIREMENTS](docs/REQUIREMENTS.md) | requirement IDs with traceability to tests |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | components, lifecycle, 12 architecture decision records |
| [DESIGN](docs/DESIGN.md) | each task's decision, checks and the mistakes they catch |
| [ICD](docs/ICD.md) | every interface: verifier, CLI, file schemas, the t03 protocol |
| [EQUATIONS](docs/EQUATIONS.md) | models, numerical methods, verification |
| [DIAGRAMS](docs/DIAGRAMS.md) | Mermaid: components, sequences, campaign |
| [THREAT_MODEL](docs/THREAT_MODEL.md) | assets, STRIDE threats, mitigations, residual risk |
| [FMEA](docs/FMEA.md) | insulin-pump FMEA behind t03; FMEA of the graders |
| [DATA](docs/DATA.md) | synthetic data generation and distributions |
| [WORKFLOWS](docs/WORKFLOWS.md) | every terminal workflow |
| [VALIDATION_REPORT](docs/VALIDATION_REPORT.md) | generated evidence |

## Reproducibility

* `uv.lock` pins every Python dependency; Docker images install from
  hash-pinned requirement files; `scripts/pin_base_images.sh` pins base images
  by digest.
* Seeds expand through `numpy.random.SeedSequence`; maths libraries run
  single-threaded; verdict JSON is key-sorted with rounded floats.
* CI grades the same submission twice and fails unless the verdicts are
  byte-identical. Validation results carry provenance and a hash manifest.

## Known limitations

* Synthetic physiology only (see [docs/DATA.md](docs/DATA.md)).
* Docker images, Harbor runs and GitHub Actions are written but not executed
  in the build environment used so far (no Docker Hub access). Everything else
  (tests, grading, privilege drop, limits, red team, determinism) has been run,
  with a local interpreter that mirrors the agent image.
* The t02 Julia port has not been run (Julia could not be installed here); CI
  has a job for it.
* GPU numbers come from running the notebook; none are claimed until then.
* Public `dev_seeds.txt` splits are not held out; use `private_seeds.txt`.
* Not a medical device. Not for clinical use.

## License

MIT
