# Interface control document

Every interface a submission, an operator or a harness touches. A change to
anything here is a breaking change and bumps the task's `version` (and
`task_version` in `task.toml`, checked by `tests/unit/test_task_layout.py`).

## 1. Verifier contract (Harbor)

| item | value |
|---|---|
| entry point | `/tests/test.sh` (bash, `set -euo pipefail`) |
| submission | artifact copied from the agent container (`artifacts` in `task.toml`) |
| seeds | `/tests/private_seeds.txt` if non-empty, else `/tests/dev_seeds.txt` |
| reward | `/logs/verifier/reward.txt` (`0` or `1`), `reward.json`, `verdict.json` |
| fail closed | `reward.txt` is written as `0` before grading starts |
| network | `no-network` for agent and verifier |
| submission user | `sandbox` (no home, nologin), interpreter `/usr/local/bin/python` |

`reward.json`: `{"reward": 0.0|1.0, "check_pass_fraction", "cases_passed",
"cases_total"}`; only `reward` is the training signal. `verdict.json`: task
id, version and every case's checks (code, status, message, measured value,
threshold), key-sorted with rounded floats so it is byte-reproducible.

## 2. Command line (`bioeng`)

```
bioeng list
bioeng seeds new N [--out FILE]
bioeng make-inputs TASK --seed HEX --outdir DIR
bioeng check-outputs TASK --seed HEX --outdir DIR [--verdict FILE]   (single-run tasks)
bioeng grade TASK --submission PATH --seeds FILE
      [--lang python|octave|julia] [--runner local|docker]
      [--sandbox-user USER] [--trusted] [--python EXE] [--image IMAGE]
      [--reward-dir DIR] [--verdict FILE] [--workdir DIR]
```

Exit codes: `0` graded and passed, `1` graded and failed (or submission
missing), `2` usage or setup error, including the refusal to run as root
without `--sandbox-user` (override with `--trusted`, for your own code only).

## 3. Submission environment

| item | value |
|---|---|
| working directory | the run's output directory |
| environment | `PATH=/usr/local/bin:/usr/bin:/bin`, `HOME=<output dir>`, `LANG=C.UTF-8`, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, every BLAS thread variable `=1`; t05 adds `PYTHONPATH=<package parent>` |
| limits | address space (task `memory_mb`), CPU time (timeout + 5 s), file size 256 MiB, 128 processes, no core dumps |
| wall clock | task `timeout_s` per run (t03: per session, plus per-step deadlines) |
| after the run | process group killed; every process of the sandbox user killed |
| outputs | regular files only; any symbolic link fails `RUN` |
| writable | the run's own output directory only |

## 4. Task interfaces

### t01 `/app/pipeline.py`
`python pipeline.py --input ecg_raw.csv --fs 360 --outdir DIR`
(Octave: `pipeline.m ecg_raw.csv 360 DIR`).

| file | content |
|---|---|
| `ecg_filtered.csv` | header `time_s,ecg_mV`, one row per input sample |
| `rpeaks.csv` | header `sample_index,time_s`, 0-based indices, `time_s = index / fs` |
| `metrics.json` | `mean_hr_bpm`, `sdnn_ms`, `rmssd_ms`, `n_beats` |

### t02 `/app/pk_pipeline.py`
`python pk_pipeline.py --indir DIR --outdir DIR` (Octave/Julia: positional
`indir outdir`). Inputs `observations.csv` (`time_h,conc_mg_per_L`),
`problem.json`. Outputs `params.json` (`CL_L_per_h`, `V1_L`, `Q_L_per_h`,
`V2_L`, `se_log{CL,V1,Q,V2}`, `t_half_alpha_h`, `t_half_beta_h`),
`prediction.csv` (`time_h,conc_mg_per_L` on 0..48 step 0.25),
`regimen.json` (`tau_h`, `dose_mg`, `infusion_h`,
`predicted_peak_mg_per_L`, `predicted_trough_mg_per_L`).

### t03 `/app/controller.py` (line protocol)
Started once per scenario with stdin/stdout pipes. One JSON object per line
each way; diagnostics to stderr only.

```
grader -> controller: {"t_min": 35.0, "cgm_mg_dL": 142.3 | null, "basal_mU_per_min": 14.2,
                       "pump_max_mU_per_min": 120.0, "exercise_announced": false}
controller -> grader: {"insulin_mU_per_min": 18.5}
```

Deadlines: 30 s for the first reply (start-up), 2 s afterwards. A reply that
is not a JSON object, lacks a finite number, arrives late, or a controller
that exits, fails `RUN`. Rates outside [0, pump max] are saturated by the
pump and fail `ACTUATOR`. Messages arrive every 5 simulated minutes; 288 per
day scenario, 72 per correction scenario.

### t04 `/app/ablation.py`
```
python ablation.py plan --problem P.json --outdir DIR
python ablation.py simulate --problem P.json --power W --heating-time S --perfusion OMEGA --nr N --dt DT --outdir DIR
```
`plan.json`: `heating_time_s`, `power_W`, `predicted_ablation_radius_mm`,
`predicted_peak_C`. `simulation.json`: `t_center_peak_C`,
`energy_at_heating_end_J`, `ablation_radius_mm`,
`energy_budget_J{deposited,stored,perfusion,boundary}`. The grader calls
`simulate` at (N, DT) = (200,1), (400,1), (800,1), (400,2), (400,0.5).

### t05 `/app/pkmodel/` (package directory)
`python -m pkmodel.cli --params P.json --regimen R.json --out conc.csv` with
the package's parent on `PYTHONPATH`. `params.json`: `CL_mL_per_min`, `V1_L`,
`Q_L_per_h`, `V2_L`. `regimen.json`: `doses[{start_h,duration_h,amount_mg}]`,
`output{start_h,stop_h,step_h}`. Output header `time_h,conc_mg_per_L`.

## 5. Result file schemas (validation)

`results/validation/<task>/<probe>.json`: `probe`, `kind`, `n_cases`,
`n_passed`, `fail_codes`, `margins{code: {min,max,threshold}}`, `seeds`
(hex), `case_verdicts_sha256`, `assessment`, `provenance`.
`results/validation/mutation/<task>.json`, `results/validation/redteam.json`,
`results/validation/MANIFEST.sha256` (sha256 of every result file).

## 6. Seed file format
One 128-bit seed per line as 32 lower-case hex digits; `#` starts a comment.
