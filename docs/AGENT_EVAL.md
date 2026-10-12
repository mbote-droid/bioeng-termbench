# Agent evaluation

The validation campaign (`docs/VALIDATION_REPORT.md`) checks the graders.
This document covers the second layer: measuring real AI agents on the tasks,
and how to report what they do. The graders are never scored by an LLM. An
LLM may help *label* transcripts, but the reward always comes from the grader.

## Two ways to run an agent

| | Harbor | `agent_eval/harness.py` |
|---|---|---|
| isolation | the task's agent image, network off | per-attempt workspace as the `sandbox` user (when run as root), rlimits, per-command timeout; network on, host kernel shared |
| needs | Docker | Python, an API key |
| agent | any Harbor agent (e.g. Claude Code) | a minimal loop with one `bash` tool |
| use for | reported scores | development, failure analysis, cheap iteration |

Both use the same graders, run sandboxed, on the same kind of hidden seeds.

## Check the pipeline first (free)

```
make agent-baselines
```

This runs two scripted agents on every task through the same loop, tool and
grader as a real agent, with no API calls:

* **oracle** writes the reference solution through the bash tool. It must
  earn reward 1 on every task. If it doesn't, the harness or the workspace
  is broken, not the agent.
* **noop** stops at once and submits whatever the workspace holds: nothing
  for t01-t04 (`NO_SUBMISSION`), and t05's shipped buggy package, which the
  grader must reject.

Results from this repository: [results/agent_eval/SUMMARY.md](../results/agent_eval/SUMMARY.md).

## Protocol for a real run

1. **Seeds.** Iterate on the public dev seeds. For numbers you report, make
   a private seed file that is never committed (`private_seeds.txt` is
   git-ignored):

       uv run bioeng seeds new 8 --out private_seeds.txt

2. **Fixed settings.** Report them with the results: model id exactly as
   sent to the API, date, max turns (default 60), max output tokens per turn
   (8000), command timeout (600 s), token budget per attempt (default
   2,000,000), seed file (dev or private), harness commit.
3. **Attempts.** At least 3 per task and model; 5 gives usable pass@k.
   Attempts are independent: a fresh workspace, no memory of earlier ones.
4. **Run.**

       export ANTHROPIC_API_KEY=...        # environment only; never written to disk
       make agent-eval MODEL=<model-id> TASKS="t05-pk-debug" ATTEMPTS=1     # check spend first
       make agent-eval MODEL=<model-id> TASKS="t01-ecg-denoise-rpeaks t02-pk-fit-dose t03-glucose-closed-loop t04-bioheat-ablation t05-pk-debug" ATTEMPTS=3

   For reported numbers add `SEEDS=private_seeds.txt`.
5. **Summarise.** `make agent-summary` writes
   `results/agent_eval/SUMMARY.md`. Commit `results/agent_eval/`: the
   transcripts are the evidence.

## Cost

Every `summary.json` records input, cache-write, cache-read and output
tokens. Multiply by your provider's current prices to get the cost; prices
change, so none are written here. The harness keeps cost down in two ways:

* **Prompt caching.** The system prompt and tool definition, and each turn's
  conversation so far, carry cache breakpoints, so later turns read most of
  their input from the cache instead of paying full price for it.
* **A per-attempt token budget** (`BUDGET=` in make, `--token-budget` in the
  harness). An attempt that reaches it stops with `stop: token_budget` and is
  graded on whatever it has written.

Start with one attempt on t05 (the shortest task) and read its token counts
before launching a full run.

## Metrics

| metric | meaning |
|---|---|
| pass rate, 95 % Wilson interval | share of attempts with reward 1 (all checks on all hidden cases) |
| pass@k | 1 if any of the k attempts passed |
| failing codes | which checks failed; each code maps to an engineering mistake (`docs/DESIGN.md`) |
| turns, tokens, minutes | effort and cost per attempt |

With 3 attempts, a 2-of-3 pass rate has a 95 % interval of about 21-94 %.
Small samples are reported with their intervals, not as point estimates.

## Failure analysis

Read the transcripts of failed attempts and give each one a single primary
label. The failing codes say *what* broke; the transcript says *why*.

| label | typical evidence |
|---|---|
| contract misread | wrong file name or schema, units the instruction stated (`SCHEMA`, `AMPLITUDE`, `PLAN_FORMAT`) |
| numerics | unconverged or first-order solver, wrong error model (`CONVERGENCE_*`, `PARAMS_LR`, `PRED_ACCURACY`) |
| unsafe decision | a plan or controller that works on average but not for every patient (`HYPO`, `REGIMEN_SAFE`, `SAFETY`) |
| over-caution | safe but fails the performance limits (`TIME_IN_RANGE`, `EFFICIENCY`) |
| incomplete debugging | t05: found the reported bugs, missed the silent ones |
| gave up or ran out | stopped early, hit max turns or the token budget |
| gaming attempt | tried to read grader files, fake outputs or detect the test harness |

A gaming attempt is reported even if it failed: it shows what an agent will
try when the reward is at stake, and it tests the sandbox.

## Reporting template

```
Model: <id>    Date: <yyyy-mm-dd>    Harness: <commit>    Seeds: private (8 per task)
Settings: 60 turns, 8000 tokens/turn, 600 s/command, 2 M tokens/attempt

<table from results/agent_eval/SUMMARY.md>

Failure labels (failed attempts): numerics 4, unsafe decision 2, ...
Notable transcripts: results/agent_eval/<model>/<task>/attempt-<k>/transcript.jsonl
```
