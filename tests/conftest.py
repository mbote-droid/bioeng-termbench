from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO / "tasks"


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO
