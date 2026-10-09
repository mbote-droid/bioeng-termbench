# Architecture

## Context

BioEng-TermBench produces **tasks** (what an AI agent is asked to do, in the
Harbor format) and **graders** (code that decides, from hidden cases, whether
the agent's work is correct). The graders are the product: in reinforcement
learning they are the reward signal, so a grader defect becomes a learned
behaviour. The architecture is organised around one question: how do we know
a grader is right, and how do we keep it right as it changes?

```
            agent container (Harbor)                verifier container (Harbor)
      +---------------------------------+     +-------------------------------------------+
      | instruction.md, example inputs  |     | /tests (0700): seeds, test.sh             |
      | numpy/scipy (hash-pinned)       |     | /opt/grader (0700): bioengbench, pint      |
      | agent writes /app/<submission>  | --> | runs submission as `sandbox` with limits  |
      +---------------------------------+     | writes /logs/verifier/reward.{txt,json}   |
                                              +-------------------------------------------+
```

The agent never sees grader code, seeds or truth. The verifier runs the
submission in a separate container, as an unprivileged user, with CPU, wall,
memory, file-size and process limits, and compares outputs with truth it
regenerates from 128-bit seeds.

## Building blocks

| package | role | key types |
|---|---|---|
| `core/task.py` | fixed grading lifecycle (Template Method) | `TaskPlugin`, `RunPlan`, `Execution` |
| `core/registry.py` | plugin discovery (microkernel) | `@register` |
| `core/runner.py` | where and how a submission runs (ports and adapters) | `Runner`, `LocalRunner`, `DockerRunner`, `Session`, `RunSpec` |
| `core/seeds.py` | 128-bit seeds, independent streams | `streams(seed, n)` |
| `core/scoring.py` | results, binary reward, canonical JSON | `CheckResult`, `CaseResult`, `Verdict` |
| `checks/` | reusable checks (Strategy, Composite) | `check`, `CheckGroup`, readers, matching, units, stats |
| `models/` | physiology and physics simulators | ECG, PK, glucose, bioheat |
| `tasks/` | one module per task: case generator + check suite | `t01` .. `t05` |
| `tooling/cli.py` | the `bioeng` command | `grade`, `make-inputs`, `check-outputs`, `seeds` |
| `validation/` | evidence that graders are right | probes, campaign, mutation, red team, report |
| `gpu/` | 3D solver and CPU/GPU benchmark (outside grading) | `bioheat3d.py` |

Dependency rule: `tasks` depend on `core`, `checks` and `models`; `core`
depends on nothing task-specific; `models` depend only on numpy/scipy.
Nothing in `src/` imports `validation/` or `gpu/`.

## The grading lifecycle

```
grade(submission, seeds, runner):
  for seed in seeds:
      case   = task.make_case(seed)                 # truth, from the seed only
      task.write_inputs(case, input_dir)            # what the agent may see
      exec   = task.execute(case, staged, runner)   # one or more runs (RunPlans)
      result = task.checks().evaluate(ctx)          # gated, composite checks
  reward = 1 if every check passed on every case else 0
```

Tasks fill in steps; they never touch sandboxing or scoring. Three
extensions of the basic lifecycle were needed and are part of the contract:

* **Run plans** (t04, t05): several runs per case with extra arguments, each
  in its own output directory. t04 reruns the agent's solver at five
  resolutions; t05 runs five regimens.
* **Closed loop** (t03): `execute` is overridden. The grader keeps the
  patient in its own process and drives the controller through a
  JSON-line `Session` with per-step deadlines.
* **Directory submissions** (t05): a package is staged and put on
  `PYTHONPATH` through `RunSpec.extra_env`.

## Architecture decision records

Each ADR states the decision, the alternatives, and the evidence.

### ADR-001 Binary reward over all checks and all hidden cases
Partial credit rewards a solution that is right on average and wrong on
some patients; in a medical setting that is the failure that matters. The
fraction of checks passed is still reported for analysis. *Consequence:*
negative controls are judged by P(pass all 8 cases), not per case.

### ADR-002 Graders validated like instruments
A grader is a measuring instrument for correctness, so it gets a
calibration record: independent references (different methods and
languages) must pass, realistic mistakes must fail with the right code,
each check must be necessary (mutation testing), and the sandbox must hold
against attacks. Thresholds come from physics or clinical practice, never
from the reference's own error. *Evidence:* `docs/VALIDATION_REPORT.md`.

### ADR-003 Truth regenerated from seeds, never stored
Hidden cases are a pure function of a 128-bit seed
(`numpy.random.SeedSequence`), so the verifier needs only the seed list and
the grader code, and any verdict can be reproduced exactly. Rejected:
shipping hidden data files (larger attack surface, harder to rotate).

### ADR-004 Separate verifier container and unprivileged sandbox user
The submission runs as `sandbox` in a container without the grader's venv,
seeds or truth. *Evidence:* the red team shows that without the sandbox user
a t03 controller forges its own simulation record and earns a passing
verdict; with it, every attack is contained. The CLI therefore refuses to
grade as root without `--sandbox-user`.

### ADR-005 Kill by process group and by user id
Killing the process group on timeout is not enough: a submission can call
`setsid()` and double-fork. After every run the runner also kills every
process owned by the sandbox user, and caps that user at 128 processes.
*Consequence:* one grading process per sandbox user at a time.

### ADR-006 Exact solutions in the graders wherever the physics allows
PK uses the matrix exponential (no step size, no tolerance, no stiffness
issue); the ECG z-equation uses an exact exponential integrator; bioheat is
verified against exact conduction and steady Pennes solutions. Where no
exact solution exists (t03 RK4, t04 FV), the grader's solver is verified
against tighter solvers and by observed order.

### ADR-007 Verify the agent's solver by its order of accuracy (t04)
Comparing one output with a reference cannot tell a converged second-order
solver from a first-order one tuned to the test case. Rerunning the
agent's code at refined resolutions measures the order directly. Space and
time are refined separately because joint refinement let the error terms
cancel (observed order 0.76 in one reference case of 60), and the order is
measured on the stored energy, a volume integral, because the centre
temperature of a correct finite-element solver converged irregularly on
these grids (orders 0.36 and 1.49 in two of eight cases).

### ADR-008 Smooth perfusion law (t04)
Damage-driven vascular shutdown makes the perfusion front sub-millimetre and
the measured order about 1.5 even for a correct scheme. A logistic
temperature law keeps the physics and makes order verification possible.
Recorded in `models/bioheat.py` and `docs/EQUATIONS.md`.

### ADR-009 Closed-loop testing as software in the loop (t03)
A controller judged on a recorded trace can be tuned to the trace. The
grader owns the plant and the sensor; the controller sees only what a pump
sees. Simulation records are written after the controller's processes are
gone, into a directory the sandbox user cannot write.

### ADR-010 Robust design judged at the truth (t02, t04)
The agent designs under uncertainty (estimated PK parameters; perfusion
range); safety is judged at the true parameters (t02) or across the whole
range (t04), and efficiency against the true optimum. A plan that is only
right for its own model fails.

### ADR-011 Reproducibility as a requirement, proven in CI
`uv.lock`, hash-pinned image requirements, digest-pinned base images,
single-threaded BLAS, canonical verdict JSON; CI grades twice and compares
bytes. Validation results carry provenance and a SHA-256 manifest.

### ADR-012 Multi-language references
Python and Octave references (and an unverified Julia port of t02) show the
contract is language-neutral and catch graders that accidentally depend on a
Python idiom. The Octave closed-loop controller found a real pitfall:
`fgetl(stdin)` blocks on an open pipe; `input('', 's')` does not.

## Quality attributes and how they are met

| attribute | mechanism | evidence |
|---|---|---|
| validity | references, negative controls, boundary probes, mutation testing | validation report |
| security | separate container, sandbox user, limits, reaping, symlink rejection | red team |
| reproducibility | seeds, locks, single thread, canonical JSON | CI byte comparison, determinism tests |
| extensibility | plugin registry, Template Method, run plans | five tasks, three execution styles |
| performance | exact solvers, vectorised checks, caching of truth per seed | t04 verifier about 25 s per case |
| traceability | requirement IDs mapped to tests | `docs/REQUIREMENTS.md` |
