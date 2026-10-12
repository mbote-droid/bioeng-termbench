"""The agent harness end to end with a scripted stand-in for the API: the
"agent" copies the oracle solution into its workspace, then stops; the real
grader must then award reward 1. No network, no API key."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("harness", ROOT / "agent_eval/harness.py")
harness = importlib.util.module_from_spec(spec)
sys.modules["harness"] = harness
spec.loader.exec_module(harness)


class Block(SimpleNamespace):
    def model_dump(self, **_):
        return {k: v for k, v in vars(self).items()}


class ScriptedClient:  # the harness has its own; this one records usage
    """Plays back a fixed list of bash commands, then ends the turn."""

    def __init__(self, commands):
        self.commands = list(commands)
        self.messages = self

    def create(self, **kw):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        if self.commands:
            cmd = self.commands.pop(0)
            block = Block(
                type="tool_use", id=f"t{len(self.commands)}", name="bash", input={"command": cmd}
            )
            return SimpleNamespace(content=[block], usage=usage, stop_reason="tool_use")
        return SimpleNamespace(
            content=[Block(type="text", text="done")], usage=usage, stop_reason="end_turn"
        )


def _args(**kw):
    base = dict(max_turns=10, max_tokens=100, command_timeout=60.0)
    return argparse.Namespace(**(base | kw))


def test_workspace_rewrites_app_paths_and_hides_the_grader(tmp_path) -> None:
    app, instruction = harness.prepare_workspace("t05-pk-debug", tmp_path)
    assert " /app/" not in instruction and "`/app/" not in instruction and str(app) in instruction
    assert (app / "pkmodel" / "cli.py").exists()
    assert not (app / "Dockerfile").exists()


def test_scripted_agent_with_the_oracle_earns_reward(tmp_path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    work.chmod(0o755)
    app, instruction = harness.prepare_workspace("t03-glucose-closed-loop", work)
    oracle = ROOT / "tasks/t03-glucose-closed-loop/solution/controller.py"
    client = ScriptedClient(
        [f"cat > controller.py <<'PYEOF'\n{oracle.read_text()}\nPYEOF", "ls -l controller.py"]
    )
    log_path = tmp_path / "t.jsonl"
    with log_path.open("w") as log:
        stats = harness.agent_loop(client, "scripted", instruction, app, log, _args())
    assert stats["stop"] == "end_turn" and stats["turns"] == 3
    assert any(
        "controller.py" in json.loads(line).get("output", "")
        for line in log_path.read_text().splitlines()
    )
    out = tmp_path / "out"
    out.mkdir()
    seeds = tmp_path / "seeds.txt"
    seeds.write_text(
        "\n".join(
            (ROOT / "tasks/t03-glucose-closed-loop/tests/dev_seeds.txt")
            .read_text()
            .splitlines()[:3]
        )
    )
    result = harness.grade("t03-glucose-closed-loop", app / "controller.py", seeds, out)
    assert result == {"reward": 1, "failing_codes": []}, (out / "grade.log").read_text()


def test_command_timeout_is_enforced(tmp_path) -> None:
    out = harness.run_bash("sleep 5", tmp_path, 0.5)
    assert "timed out" in out


@pytest.mark.parametrize("n", [harness.OUTPUT_LIMIT * 2])
def test_long_output_is_truncated(tmp_path, n) -> None:
    out = harness.run_bash(f"python3 -c \"print('x' * {n})\"", tmp_path, 30)
    assert len(out) <= harness.OUTPUT_LIMIT + 100


def test_cache_breakpoint_marks_only_the_last_block_and_copies() -> None:
    history = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "x"}]},
    ]
    sent = harness.with_cache_breakpoint(history)
    assert sent[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in history[-1]["content"][-1]  # history untouched
    assert sum("cache_control" in str(m) for m in sent) == 1
    first = harness.with_cache_breakpoint(history[:1])
    assert first[0]["content"] == [
        {"type": "text", "text": "task", "cache_control": {"type": "ephemeral"}}
    ]


class CountingClient(ScriptedClient):
    def create(self, **kw):
        resp = super().create(**kw)
        resp.usage = SimpleNamespace(input_tokens=600, output_tokens=0)
        return resp


def test_token_budget_stops_the_attempt(tmp_path) -> None:
    client = CountingClient(["true"] * 20)
    with (tmp_path / "t.jsonl").open("w") as log:
        stats = harness.agent_loop(
            client, "scripted", "task", tmp_path, log, _args(token_budget=1000)
        )
    assert stats["stop"] == "token_budget" and stats["turns"] == 2
    assert stats["total_tokens"] == 1200


@pytest.mark.parametrize(("agent", "reward"), [("oracle", 1), ("noop", 0)])
def test_scripted_baselines_on_t05(tmp_path, monkeypatch, agent, reward) -> None:
    """oracle writes the fixed package through the tool and passes; noop
    submits the shipped buggy package, which the grader must reject."""
    monkeypatch.setattr(harness, "RESULTS", tmp_path / "results")
    seeds = tmp_path / "seeds.txt"
    seeds.write_text(
        "\n".join((ROOT / "tasks/t05-pk-debug/tests/dev_seeds.txt").read_text().splitlines()[:2])
    )
    s = harness.attempt(
        "t05-pk-debug", f"baseline-{agent}", 1, _args(agent=agent, seeds=str(seeds))
    )
    assert s["reward"] == reward, s
    assert (tmp_path / "results" / f"baseline-{agent}" / "t05-pk-debug" / "attempt-1").is_dir()
    if agent == "noop":
        assert s["failing_codes"] and s["turns"] == 1
    assert "| baseline-" in harness.summarise()


def test_output_limit_is_configurable(tmp_path) -> None:
    out = harness.run_bash("python3 -c \"print('x' * 5000)\"", tmp_path, 30, limit=1000)
    assert len(out) <= 1100 and "[truncated]" in out
