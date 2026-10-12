# Changelog

Versions follow semantic versioning. Each task also carries its own
`task_version` in `task.toml`, bumped whenever its grader changes, so a
reward is always tied to the grader that gave it.

## Unreleased

* Agent harness: OpenAI-compatible providers with free tiers (Groq, Gemini,
  OpenRouter, Cerebras) through `--provider` / `PROVIDER=`, standard library
  only, with request spacing and retries for rate limits. Attempts stopped by
  a provider error are recorded as `api_error` and left out of pass rates.
* `--output-limit` / `OUTPUT_LIMIT=` caps each command's output shown to the
  agent, to save tokens on free tiers.

## 1.0.0 (2026-10-12)

First public release.

* Five tasks in the Harbor format: t01 ECG denoising, beat detection and HRV
  (task 2.0.0); t02 pharmacokinetic fitting and dosing; t03 closed-loop
  insulin delivery; t04 thermal ablation planning; t05 debugging a
  pharmacokinetics package (all 1.0.0).
* Sandboxed grading: unprivileged submission user, resource limits, process
  group and user-id reaping, symlink rejection, refusal to grade as root
  without isolation.
* Grader validation campaign (Snakemake): 13 reference solutions with 0 false
  rejects on 820 hidden cases; 33 of 33 realistic wrong solutions rejected;
  44 of 45 checks necessary under mutation; 7 of 7 sandbox attacks contained.
  Report: `docs/VALIDATION_REPORT.md`.
* 3D GPU bioheat benchmark (NumPy, CuPy, fused CUDA kernel), verified against
  the 1D grader model, with measured Tesla T4 results.
* Agent evaluation harness with prompt caching, a per-attempt token budget
  and scripted oracle and no-op baselines.
* Documentation: requirements with traceability, architecture with 12
  decision records, design, interfaces, equations, diagrams, threat model,
  FMEA, data, workflows, agent evaluation protocol and a technical report.
