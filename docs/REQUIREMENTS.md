# Requirements and traceability

Each requirement has an ID, a rationale, the design element that meets it,
and the verification that proves it. "Campaign" means the validation
campaign (`make validate`, results in `docs/VALIDATION_REPORT.md`).

## Stakeholders

* **Model trainers** need rewards that are correct, hard to game and
  reproducible.
* **Task authors** need a framework where adding a task cannot weaken
  isolation or scoring.
* **Reviewers** need evidence, not claims: every number traceable to a
  result file.

## Functional

| ID | requirement | rationale | design | verification |
|---|---|---|---|---|
| F-01 | Every task is a valid Harbor task (instruction, task.toml 1.3, agent and verifier images, oracle solution, verifier script) | runs in standard harnesses | `tasks/<id>/` | `tests/unit/test_task_layout.py::test_harbor_layout` |
| F-02 | Hidden cases are generated from 128-bit seeds only | no stored truth; reproducible | `core/seeds.py`, `make_case(seed)` | `tests/unit/test_seeds.py`; `test_task_layout::test_dev_seeds_are_128_bit_and_distinct` |
| F-03 | Reward is 1 only if every check passes on every case | partial correctness is not safe in medicine | `core/scoring.py` | `tests/unit/test_scoring.py` |
| F-04 | Every failure is reported with a diagnostic code, measured value and threshold | debuggable rewards; analysis | `checks/base.py` | negative-control tests in `tests/tasks/` |
| F-05 | Graders check physics: units, conservation, convergence order, safety at the truth, robustness | the point of the benchmark | task modules | per-check negative controls; campaign |
| F-06 | Python and Octave submissions are accepted where the task declares them | language-neutral contracts | `TaskPlugin.argv(lang)`, `--lang` | Octave references in `tests/tasks/test_t0*.py` |
| F-07 | Closed-loop tasks run the submission as a live controller with per-step deadlines | software-in-the-loop testing | `core/runner.py::Session`, `t03` | `tests/tasks/test_t03.py` protocol tests |
| F-08 | Multi-run tasks rerun the submission with different arguments | order-of-accuracy and scenario testing | `RunPlan` | `tests/tasks/test_t04.py`, `test_t05.py` |
| F-09 | Package (directory) submissions | debugging tasks need a code base | `stage_submission`, `extra_env` | `tests/tasks/test_t05.py` |

## Validity

| ID | requirement | rationale | design | verification |
|---|---|---|---|---|
| V-01 | Correct solutions using other methods or languages pass every case; the 95 % upper bound on the per-case false-reject rate is reported (3/n with no failures: 1 % for t03 at 300 cases, 3 % at 100, 10 % for t04 at 30, which is compute-limited) | wrong rejections punish correct behaviour | thresholds from physics, not from the reference | campaign: reference probes |
| V-02 | Each realistic mistake fails with its expected code, and its probability of earning reward over 8 hidden cases is below 5 % (95 % upper bound) | false accepts train reward hacking | negative controls and mutants | campaign: negative probes |
| V-03 | Every check is necessary: disabling it makes a test fail | no dead checks | tests per check | campaign: grader mutation (`check_off`) |
| V-04 | Grader models are verified independently of the task | truth must be true | exact solutions, LSODA, observed order, FEM, Octave, 3D | `tests/unit/test_*_model.py`, `test_gpu_bioheat3d.py` |
| V-05 | Defensible methods at or beyond a contract limit fail only on that clause, never on another | the contract, not the grader, decides | boundary probes | campaign: t01 `median_baseline` |
| V-06 | Each threshold's justification is written next to it | reviewable | comments in task modules | review |

## Security

| ID | requirement | design | verification |
|---|---|---|---|
| S-01 | A submission cannot read seeds, grader code or truth | 0700 paths, separate interpreter, sandbox user | red team `t01_escape` |
| S-02 | A submission cannot write outside its own output directory | root-owned case directories | `t01_escape`, `t03_forge` |
| S-03 | CPU, wall clock, memory, file size and process count are limited | `rlimit`s, timeouts | `t01_memory`, `t01_disk_fill`, `t01_fork_bomb`, `tests/unit/test_runner.py` |
| S-04 | No process of a submission survives its run | kill process group and sandbox uid | `t01_daemon`, `test_runner.py::test_detached_daemon_is_reaped_by_user_id` |
| S-05 | Outputs are regular files | symlink rejection in `TaskPlugin.execute` | `t01_symlink` |
| S-06 | The CLI refuses to run a submission as root without isolation | `_check_isolation` | `tests/unit/test_cli_isolation.py` |
| S-07 | The agent image contains no grader material | separate images | `test_task_layout::test_agent_image_holds_no_grader_material` |
| S-08 | Instructions carry unique canary GUIDs | contamination detection | `test_task_layout::test_canaries_are_unique` |

## Reproducibility

| ID | requirement | design | verification |
|---|---|---|---|
| R-01 | Dependencies locked: Python (uv.lock), images (hash-pinned requirements, digest-pinned bases) | `uv.lock`, `docker/*.txt`, `scripts/pin_base_images.sh` | `uv sync --frozen` in CI |
| R-02 | Same submission and seeds give a byte-identical verdict | single-threaded BLAS, hash seed, canonical JSON | determinism tests (t01, t03); CI `reproducibility` job |
| R-03 | Every workflow is one command | `Makefile`, `bioeng` CLI, Snakemake | CI runs the same targets |
| R-04 | Every validation number is traceable to a result file with provenance; files hashed | `validation/provenance.py`, `MANIFEST.sha256` | report generator |

## Performance

| ID | requirement | verification |
|---|---|---|
| P-01 | Verifier finishes within `task.toml` limits for 8 hidden cases (t04 is the slowest at about 25 s per case) | campaign timings (`seconds` in result files) |
| P-02 | The GPU benchmark verifies correctness before timing | `tests/unit/test_gpu_bioheat3d.py`, notebook gates |

## Traceability: requirements to tests

| test file | requirements |
|---|---|
| `tests/unit/test_task_layout.py` | F-01, F-02, S-07, S-08 |
| `tests/unit/test_seeds.py`, `test_scoring.py` | F-02, F-03, R-02 |
| `tests/unit/test_runner.py` | S-03, S-04, F-07 |
| `tests/unit/test_cli_isolation.py` | S-06 |
| `tests/unit/test_ecg_model.py`, `test_pk_model.py`, `test_glucose_model.py`, `test_bioheat_model.py`, `test_gpu_bioheat3d.py` | V-04, P-02 |
| `tests/unit/test_checks_base.py`, `test_matching.py`, `test_units.py`, `test_stats.py` | F-04, F-05 |
| `tests/tasks/test_t01_*.py` .. `test_t05.py` | F-05 .. F-09, V-02, V-05, R-02 |
| `validation/` (campaign, mutation, red team) | V-01 .. V-03, S-01 .. S-05, R-04, P-01 |
