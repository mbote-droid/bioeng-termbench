# Terminal workflows

Every workflow in this project is one command, runs the same way on a laptop
and in CI, and leaves a record. This page lists them in the order a new
contributor meets them.

## 1. Environment

```
uv sync --frozen            # exact environment from uv.lock (Python 3.12)
uv run bioeng list          # registered tasks and grader versions
```

`--frozen` fails instead of silently re-resolving: a changed dependency is a
reviewed change to `uv.lock`, never a side effect.

## 2. Day-to-day checks

```
make lint          # ruff (lint + format), shellcheck on every task script
make typecheck     # mypy --strict on the package
make test-fast     # unit tests + negative controls (about 2 min)
make test          # everything, including references on dev seeds (Octave if installed)
make check         # lint + typecheck + test: what CI runs
```

## 3. Working on a task

```
uv run bioeng seeds new 8 --out /tmp/seeds.txt
uv run bioeng make-inputs t04-bioheat-ablation --seed <hex> --outdir /tmp/in   # what the agent sees
uv run bioeng grade t04-bioheat-ablation \
    --submission tasks/t04-bioheat-ablation/solution/ablation.py \
    --seeds tasks/t04-bioheat-ablation/tests/dev_seeds.txt --verdict /tmp/verdict.json
```

As root, grading refuses to run without isolation; use the verifier's
configuration:

```
sudo useradd --system --no-create-home --shell /usr/sbin/nologin sandbox     # once
uv run bioeng grade ... --sandbox-user sandbox --python /opt/agent-python/bin/python
```

`/opt/agent-python` mirrors the agent image's interpreter (numpy, scipy,
pytest from `docker/agent-requirements.txt`, no grader):

```
uv venv /opt/agent-python --python 3.12
uv pip install --python /opt/agent-python/bin/python --require-hashes -r docker/agent-requirements.txt
```

Octave submissions: `--lang octave`. Interactive (t03) and multi-run (t04,
t05) tasks need no extra flags.

## 4. Validating a grader change

```
make validate-quick          # probes + grader mutation, in parallel (JOBS=4)
sudo make redteam            # attacks on the sandbox (root + sandbox user)
make validate                # all of it, then docs/VALIDATION_REPORT.md
```

Snakemake reruns only what a change invalidates: editing
`src/bioengbench/tasks/t03_glucose.py` reruns t03's probes and mutation
tests; editing a reference reruns that probe. Results land in
`results/validation/`, each with provenance (git commit, `uv.lock` hash,
package versions, seeds), and `MANIFEST.sha256` hashes them all.

## 5. Reproducing a verdict

```
uv run bioeng grade t01-ecg-denoise-rpeaks --submission ... --seeds ... --verdict a.json
uv run bioeng grade t01-ecg-denoise-rpeaks --submission ... --seeds ... --verdict b.json
cmp a.json b.json            # byte-identical; CI does exactly this
```

## 6. Images

```
make build                   # wheel for the grader image
make images                  # agent and grader base images (Docker)
scripts/pin_base_images.sh   # replace tags by immutable digests
```

## 7. GPU benchmark (outside grading)

```
make gpu-bench               # CPU parts anywhere; GPU parts with CuPy + CUDA
```

or open `gpu/BioEng_GPU_Benchmark.ipynb` in Colab (free T4). The benchmark
refuses to report speed before the correctness gates pass.

## 8. Harbor

```
harbor run -p tasks/t03-glucose-closed-loop -a oracle     # oracle must score 1
```

The oracle (`solution/solve.sh`) must earn reward 1 and an empty submission
reward 0; `tests/test.sh` writes reward 0 first, so any crash fails closed.

## 9. Agent evaluation

```
make agent-baselines                                           # scripted oracle and no-op agents; no API key
export GEMINI_API_KEY=...                                      # environment only, never written to disk
make agent-eval PROVIDER=gemini MODEL=gemini-flash-latest TASKS="t05-pk-debug" ATTEMPTS=1      # smoke test
make agent-eval PROVIDER=gemini MODEL=gemini-flash-latest TASKS="t01-ecg-denoise-rpeaks t05-pk-debug" ATTEMPTS=3 SEEDS=private_seeds.txt
make agent-eval MODEL=<anthropic-model-id> TASKS="t05-pk-debug"       # with ANTHROPIC_API_KEY instead
make agent-summary                                             # -> results/agent_eval/SUMMARY.md
```

`PROVIDER=` is `anthropic` (default), `groq`, `gemini`, `openrouter` or
`cerebras`; `OUTPUT_LIMIT=` caps each command's output shown to the agent
(characters, default 20000); `BUDGET=` sets the per-attempt token budget (default 2,000,000). Protocol and
reporting: `docs/AGENT_EVAL.md`.

## 10. Releasing

1. Update `CHANGELOG.md` and the version in `pyproject.toml`,
   `src/bioengbench/__init__.py`, `CITATION.cff` and `uv.lock`
   (`uv lock` after editing `pyproject.toml`).
2. `make check`, then commit and push; wait for CI to pass.
3. Tag and publish: `gh release create v1.0.0 --title "v1.0.0" --notes-file CHANGELOG.md`.
   With the Zenodo GitHub integration switched on, Zenodo archives the release
   and mints a DOI (metadata from `.zenodo.json`).
