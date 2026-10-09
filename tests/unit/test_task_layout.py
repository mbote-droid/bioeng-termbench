"""Every task directory follows the Harbor contract and matches its plugin."""

import re
import tomllib
from pathlib import Path

import pytest

from bioengbench.core import registry
from bioengbench.core.seeds import read_seed_file

TASKS = Path(__file__).resolve().parents[2] / "tasks"
TASK_DIRS = sorted(p for p in TASKS.iterdir() if p.is_dir())


def test_every_task_directory_has_a_plugin_and_vice_versa() -> None:
    assert sorted(p.name for p in TASK_DIRS) == registry.task_ids()


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_harbor_layout(task_dir: Path) -> None:
    for rel in (
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "solution/solve.sh",
        "tests/test.sh",
        "tests/Dockerfile",
        "tests/dev_seeds.txt",
    ):
        assert (task_dir / rel).is_file(), rel
    meta = tomllib.loads((task_dir / "task.toml").read_text())
    assert meta["schema_version"] == "1.3"
    assert meta["task"]["name"] == f"bioeng-termbench/{task_dir.name}"
    assert meta["metadata"]["task_version"] == registry.get(task_dir.name).version
    assert meta["agent"]["network_mode"] == meta["verifier"]["network_mode"] == "no-network"
    assert meta["verifier"]["environment_mode"] == "separate"
    assert meta["artifacts"], "the verifier needs the submission as an artifact"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_instruction_carries_a_unique_canary(task_dir: Path) -> None:
    text = (task_dir / "instruction.md").read_text()
    assert re.search(r"harbor-canary GUID [0-9a-f-]{36}", text)


def test_canaries_are_unique() -> None:
    guids = [
        re.search(r"GUID ([0-9a-f-]{36})", (d / "instruction.md").read_text()).group(1)
        for d in TASK_DIRS
    ]
    assert len(set(guids)) == len(guids)


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_dev_seeds_are_128_bit_and_distinct(task_dir: Path) -> None:
    seeds = read_seed_file(task_dir / "tests/dev_seeds.txt")
    assert len(seeds) == len(set(seeds)) >= 8
    assert all(s.bit_length() > 64 for s in seeds)


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_agent_image_holds_no_grader_material(task_dir: Path) -> None:
    env = task_dir / "environment"
    for path in env.rglob("*"):
        if path.is_file():
            text = path.read_text(errors="ignore")
            assert not re.search(r"(import|from) bioengbench", text), path
            assert "dev_seeds" not in text, path
