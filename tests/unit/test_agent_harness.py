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
    def model_dump(self):
        return {k: v for k, v in vars(self).items()}


class ScriptedClient:
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
