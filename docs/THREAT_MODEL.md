# Threat model

## What is being protected

| asset | why it matters |
|---|---|
| reward integrity | a reward that can be earned without solving the task trains reward hacking |
| hidden seeds and truth | with them, any task is solved by lookup |
| grader code | reading it reveals thresholds and reference outputs |
| the verifier host | denial of service ends grading for everyone |
| reproducibility | an unreproducible verdict cannot be audited |

## Adversary

The agent: controls the submission's code and everything it writes, may
probe the environment, and is optimised against the reward. It does not
control the verifier image, the seeds, or the harness. Model: an
unprivileged local attacker inside the verifier container, without network.

## Trust boundaries

```
agent container | verifier container: test.sh (root) -> bioeng grade (root, /opt/grader 0700)
                |                       -> submission (uid sandbox, limits, no network)
```

## Threats, mitigations, evidence (STRIDE)

| # | threat | STRIDE | mitigation | evidence |
|---|---|---|---|---|
| T1 | read seeds or truth (`/tests`, grader venv, other cases) | I | `/tests` and `/opt/grader` mode 0700; truth never written to disk; sandbox user | red team `t01_escape` |
| T2 | import the generator to recompute truth | I | grader venv absent from the submission interpreter | `t01_escape` (`import_grader` fails) |
| T3 | forge grader records (t03 simulation, outputs of other runs) | T | grader-owned directories not writable by `sandbox`; t03 records written after the controller is dead | `t03_forge`: contained sandboxed; **earns a pass unsandboxed** |
| T4 | tamper with inputs or the staged submission | T | inputs and submission root-owned, read-only to `sandbox` | `t01_escape` |
| T5 | symbolic-link outputs (point the grader at other files) | T, I | any symlink in an output directory fails `RUN` | `t01_symlink` |
| T6 | survive the run (background child, `setsid` daemon) and change outputs later | T | kill process group after every run; kill every process of the sandbox uid | `t01_daemon`, `test_runner.py` |
| T7 | fork bomb | D | `RLIMIT_NPROC` = 128 for the sandbox user | `t01_fork_bomb` (stopped at 127) |
| T8 | memory exhaustion | D | `RLIMIT_AS` per task | `t01_memory` |
| T9 | disk exhaustion | D | `RLIMIT_FSIZE` 256 MiB | `t01_disk_fill` |
| T10 | hang (no reply, infinite loop) | D | wall-clock timeout per run; per-step deadlines in t03 | `slow`, `silent` probes |
| T11 | read the grader's environment (`/proc/<pid>/environ`) | I | different uid; scrubbed environment for the submission | `t01_escape` |
| T12 | network exfiltration or download of solutions | I | `network_mode = "no-network"` (Harbor) | configuration, not tested locally |
| T13 | game a weak check (e.g. outputs that fit the check, not the physics) | T | physics-based checks, negative controls, mutation testing | validation report |
| T14 | memorise public dev seeds | I | private seed file mounted at evaluation time | `test.sh` seed selection |
| T15 | grade as root without isolation by mistake | E | CLI refuses unless `--trusted` | `tests/unit/test_cli_isolation.py` |
| T16 | supply chain (unpinned packages, mutable images) | T | `uv.lock`, `--require-hashes`, digest-pinned base images | `make lint`, `scripts/pin_base_images.sh` |
| T17 | training-data contamination | I | canary GUID per instruction, unique per task | `tests/unit/test_task_layout.py` |

## Residual risks (accepted, documented)

* **Shared sandbox user.** Reaping by uid means one grading process per
  sandbox user at a time; parallel grading needs one user per worker.
* **Unsandboxed local grading.** A developer grading as themselves gives the
  submission their permissions; the CLI warns. Use `--runner docker` for
  untrusted code on a workstation.
* **Timing side channels and CPU contention** are not modelled.
* **Docker runner and Harbor images** are written but not executed in the
  build environment used so far (no Docker Hub access); local runs reproduce
  the verifier's user, interpreter and limits instead.
* **Dev seeds are public**; held-out scores need `private_seeds.txt`.

## How to rerun the red team

```
sudo useradd --system --no-create-home --shell /usr/sbin/nologin sandbox   # once
sudo uv run python validation/campaign.py redteam
```
Results: `results/validation/redteam.json` and the Red team section of
`docs/VALIDATION_REPORT.md`.
