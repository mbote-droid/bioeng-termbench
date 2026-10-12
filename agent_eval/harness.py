"""Evaluate a real AI agent on BioEng-TermBench tasks.

Two ways to run agents:

1. Harbor (faithful; needs Docker): the task's own agent and verifier images,
   network off.
       harbor run -p tasks/t03-glucose-closed-loop -a claude-code -m <model>
2. This harness (no Docker; for development): a minimal tool-use agent loop
   against the Anthropic API. The agent gets one tool, `bash`, running in a
   per-attempt workspace that stands in for /app, as the unprivileged
   `sandbox` user when run as root, with a per-command timeout and resource
   limits. When the agent stops (or hits the turn limit) its submission is
   graded by the real grader, sandboxed, on the chosen seeds.

       export ANTHROPIC_API_KEY=...          # read from the environment, never written to disk
       uv run --with 'anthropic>=1.12.1,<2' python agent_eval/harness.py \
           --task t05-pk-debug --model <model-id> --attempts 3
       uv run python agent_eval/harness.py --summarise

Two scripted baselines run the same pipeline without an API key, so the
harness itself is checked before any money is spent:

       uv run python agent_eval/harness.py --agent oracle --task t03-glucose-closed-loop
       uv run python agent_eval/harness.py --agent noop --task t05-pk-debug

`oracle` writes the reference solution through the bash tool and must earn
reward 1. `noop` stops at once, so it submits whatever the workspace already
holds (nothing, or t05's buggy package) and must earn 0.

Cost control: the system prompt and the growing conversation are sent with
prompt-cache breakpoints, and `--token-budget` stops an attempt once its total
tokens (input, cache reads and writes, output) pass the budget.

Differences from Harbor, stated rather than hidden: /app paths in the
instruction are rewritten to the workspace; the network is not disabled; the
agent shares the host kernel. Use Harbor for reported scores; use this to
iterate quickly.

Output per attempt, under results/agent_eval/<model>/<task>/attempt-<k>/:
transcript.jsonl (every message and tool result), the submission, the
grader's verdict.json and grade.log, and summary.json (reward, failing codes,
turns, tokens, seconds). `--summarise` writes results/agent_eval/SUMMARY.md.
Protocol and reporting: docs/AGENT_EVAL.md.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pwd
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "agent_eval"
SANDBOX = "sandbox"
AGENT_PYTHON = Path("/opt/agent-python/bin/python")

SUBMISSION = {  # task -> submission path inside /app
    "t01-ecg-denoise-rpeaks": "pipeline.py",
    "t02-pk-fit-dose": "pk_pipeline.py",
    "t03-glucose-closed-loop": "controller.py",
    "t04-bioheat-ablation": "ablation.py",
    "t05-pk-debug": "pkmodel",
}

SYSTEM = (
    "You are an expert engineer working in a Linux terminal. You have one tool, `bash`, "
    "which runs a command in your working directory and returns its output. Work step by "
    "step: inspect files, write code, run and test it. When the task is complete, reply "
    "with a short summary and no tool call."
)
BASH_TOOL = {
    "name": "bash",
    "description": "Run a bash command in the task workspace. Output is truncated to 20000 characters.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "the command to run"}},
        "required": ["command"],
    },
}
OUTPUT_LIMIT = 20_000
CACHE = {"type": "ephemeral"}
BASELINES = ("oracle", "noop")


class _Block(SimpleNamespace):
    """Stand-in for an API content block (scripted baselines)."""

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(vars(self))


class ScriptedClient:
    """Plays back a fixed list of bash commands through the tool, then ends
    the turn. Same interface as `anthropic.Anthropic().messages`."""

    def __init__(self, commands: list[str]) -> None:
        self.commands = list(commands)
        self.messages = self

    def create(self, **_: Any) -> SimpleNamespace:
        usage = SimpleNamespace(input_tokens=0, output_tokens=0)
        if self.commands:
            block = _Block(
                type="tool_use",
                id=f"call-{len(self.commands)}",
                name="bash",
                input={"command": self.commands.pop(0)},
            )
            return SimpleNamespace(content=[block], usage=usage, stop_reason="tool_use")
        return SimpleNamespace(
            content=[_Block(type="text", text="done")], usage=usage, stop_reason="end_turn"
        )


def oracle_commands(task: str) -> list[str]:
    """Bash commands that write the reference solution into the workspace,
    as an agent would: a base64 tarball unpacked in place, then a listing."""
    name = SUBMISSION[task]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(
            ROOT / "tasks" / task / "solution" / name,
            arcname=name,
            filter=lambda ti: None if "__pycache__" in ti.name else ti,
        )
    payload = base64.b64encode(buf.getvalue()).decode()
    return [f"rm -rf {name} && echo {payload} | base64 -d | tar xzf -", f"ls -lR {name}"]


def make_client(agent: str, task: str) -> Any:
    if agent == "oracle":
        return ScriptedClient(oracle_commands(task))
    if agent == "noop":
        return ScriptedClient([])
    import anthropic

    return anthropic.Anthropic()


def with_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of the conversation with a cache breakpoint on the last block of
    the last message, so each turn reads the previous turns from the cache.
    The stored history is not modified (at most 4 breakpoints are allowed)."""
    last = messages[-1]
    content = last["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    content = [*content[:-1], {**content[-1], "cache_control": CACHE}]
    return [*messages[:-1], {**last, "content": content}]


def prepare_workspace(task: str, root: Path) -> tuple[Path, str]:
    """Copy what the agent image contains and rewrite /app in the instruction."""
    task_dir = ROOT / "tasks" / task
    app = root / "app"
    app.mkdir(parents=True)
    for item in (task_dir / "environment").iterdir():
        if item.name == "Dockerfile":
            continue
        dest = app / item.name
        if item.is_dir():
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(item, dest)
    instruction = (task_dir / "instruction.md").read_text().replace("/app", str(app))
    if os.geteuid() == 0:
        try:
            uid = pwd.getpwnam(SANDBOX).pw_uid
        except KeyError:
            uid = None
        if uid is not None:
            for p in [root, *root.rglob("*")]:
                os.chown(p, uid, -1)
    return app, instruction


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 20, 256 << 20))


def run_bash(command: str, cwd: Path, timeout_s: float) -> str:
    user = SANDBOX if os.geteuid() == 0 else None
    path = os.environ.get("PATH", "/usr/bin:/bin")
    if AGENT_PYTHON.exists():
        path = f"{AGENT_PYTHON.parent}:/usr/local/bin:/usr/bin:/bin"
    env = {"PATH": path, "HOME": str(cwd), "LANG": "C.UTF-8", "OMP_NUM_THREADS": "1"}
    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            user=user,
            preexec_fn=_limits,
            check=False,
        )
        out = proc.stdout
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        out += f"\n[exit {proc.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout_s:g} s]"
    if len(out) > OUTPUT_LIMIT:
        half = OUTPUT_LIMIT // 2
        out = out[:half] + "\n...[truncated]...\n" + out[-half:]
    return out


def agent_loop(
    client: Any, model: str, instruction: str, app: Path, log: Any, a: argparse.Namespace
) -> dict:
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    keys = (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
    )
    usage = dict.fromkeys(keys, 0)
    budget = getattr(a, "token_budget", 0) or 0
    turns, stop = 0, "max_turns"
    while turns < a.max_turns:
        if budget and sum(usage.values()) >= budget:
            stop = "token_budget"
            break
        turns += 1
        resp = client.messages.create(
            model=model,
            max_tokens=a.max_tokens,
            system=[{"type": "text", "text": SYSTEM, "cache_control": CACHE}],
            tools=[BASH_TOOL],
            messages=with_cache_breakpoint(messages),
        )
        for key in keys:
            usage[key] += getattr(resp.usage, key, 0) or 0
        content = [block.model_dump(exclude_none=True) for block in resp.content]
        log.write(json.dumps({"turn": turns, "role": "assistant", "content": content}) + "\n")
        messages.append({"role": "assistant", "content": content})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            stop = resp.stop_reason or "end_turn"
            break
        results = []
        for call in calls:
            output = run_bash(str(call.input.get("command", "")), app, a.command_timeout)
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
            log.write(
                json.dumps({"turn": turns, "role": "tool", "input": call.input, "output": output})
                + "\n"
            )
        messages.append({"role": "user", "content": results})
        log.flush()
    return {"turns": turns, "stop": stop, **usage, "total_tokens": sum(usage.values())}


def grade(task: str, submission: Path, seeds: Path, out: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "bioengbench.tooling.cli",
        "grade",
        task,
        "--submission",
        str(submission),
        "--seeds",
        str(seeds),
        "--verdict",
        str(out / "verdict.json"),
    ]
    if os.geteuid() == 0:
        cmd += ["--sandbox-user", SANDBOX]
    if AGENT_PYTHON.exists():
        cmd += ["--python", str(AGENT_PYTHON)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    (out / "grade.log").write_text(proc.stdout + proc.stderr)
    if not (out / "verdict.json").exists():
        return {"reward": 0, "failing_codes": ["NO_VERDICT"]}
    verdict = json.loads((out / "verdict.json").read_text())
    codes = sorted(
        {r["code"] for c in verdict["cases"] for r in c["checks"] if r["status"] == "fail"}
    )
    return {"reward": int(all(c["passed"] for c in verdict["cases"])), "failing_codes": codes}


def attempt(task: str, model: str, k: int, a: argparse.Namespace) -> dict:
    """One attempt; `model` is the API model id, or `baseline-<agent>`."""
    agent = getattr(a, "agent", "api")
    out = RESULTS / model / task / f"attempt-{k}"
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="agent-"))
    work.chmod(0o755)
    app, instruction = prepare_workspace(task, work)
    t0 = time.time()
    with (out / "transcript.jsonl").open("w") as log:
        log.write(json.dumps({"role": "user", "content": instruction}) + "\n")
        stats = agent_loop(make_client(agent, task), model, instruction, app, log, a)
    sub = app / SUBMISSION[task]
    if sub.exists():
        dest = out / "submission" / sub.name
        dest.parent.mkdir(exist_ok=True)
        if sub.is_dir():
            shutil.copytree(
                sub, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copyfile(sub, dest)
        seeds = Path(a.seeds) if a.seeds else ROOT / "tasks" / task / "tests/dev_seeds.txt"
        result = grade(task, dest, seeds, out)
    else:
        result = {"reward": 0, "failing_codes": ["NO_SUBMISSION"]}
    summary = {
        "task": task,
        "model": model,
        "agent": agent,
        "seeds": str(Path(a.seeds).name) if a.seeds else "dev_seeds.txt",
        "attempt": k,
        "seconds": round(time.time() - t0, 1),
        **stats,
        **result,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    return summary


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise() -> str:
    """Markdown table over every summary.json; also written to SUMMARY.md."""
    by: dict[tuple[str, str], list[dict]] = {}
    for p in RESULTS.rglob("summary.json"):
        r = json.loads(p.read_text())
        by.setdefault((r["model"], r["task"]), []).append(r)
    lines = [
        "| model | task | attempts | pass rate (95 % CI) | pass@k | most common failing codes | mean turns | mean tokens | mean minutes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, task), rs in sorted(by.items()):
        counts: dict[str, int] = {}
        for r in rs:
            for c in r["failing_codes"]:
                counts[c] = counts.get(c, 0) + 1
        top = (
            ", ".join(
                f"{c} x{n}" for c, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:3]
            )
            or "-"
        )
        k, n = sum(r["reward"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        turns = sum(r["turns"] for r in rs) / n
        tokens = sum(r.get("total_tokens", r["input_tokens"] + r["output_tokens"]) for r in rs) / n
        minutes = sum(r["seconds"] for r in rs) / n / 60
        lines.append(
            f"| {model} | {task} | {n} | {k / n:.0%} ({lo:.0%}-{hi:.0%}) | {int(k > 0)} | {top} "
            f"| {turns:.0f} | {tokens:,.0f} | {minutes:.1f} |"
        )
    table = "\n".join(lines) + "\n"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "SUMMARY.md").write_text(
        "# Agent evaluation results\n\nGenerated by `agent_eval/harness.py --summarise`; "
        "protocol in docs/AGENT_EVAL.md. pass@k is 1 if any attempt earned reward.\n\n" + table
    )
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", action="append", choices=sorted(SUBMISSION))
    ap.add_argument("--all-tasks", action="store_true", help="every task")
    ap.add_argument("--agent", choices=("api", *BASELINES), default="api")
    ap.add_argument("--model", help="API model id (required for --agent api)")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--seeds", help="seed file (default: the task's dev seeds)")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--token-budget", type=int, default=2_000_000, help="per attempt; 0 = none")
    ap.add_argument("--command-timeout", type=float, default=600.0)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        print(summarise(), end="")
        return 0
    tasks = sorted(SUBMISSION) if a.all_tasks else a.task
    if not tasks:
        ap.error("give --task (repeatable) or --all-tasks")
    if a.agent == "api":
        if not a.model:
            ap.error("--model is required for --agent api")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            ap.error("set ANTHROPIC_API_KEY in the environment (it is never written to disk)")
        label = a.model
    else:
        label = f"baseline-{a.agent}"
    for task in tasks:
        for k in range(1, a.attempts + 1):
            s = attempt(task, label, k, a)
            print(
                f"{task} attempt {k}: reward {s['reward']} {s['failing_codes']} ({s['turns']} turns)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
