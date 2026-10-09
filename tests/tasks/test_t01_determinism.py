"""Same submission, same seeds: byte-identical verdict (REQ: determinism)."""

import sys

import pytest
from conftest import T01

from bioengbench.core.runner import LocalRunner


@pytest.mark.slow
def test_verdict_is_byte_identical_across_runs(t01, t01_dev_seeds, tmp_path) -> None:
    sub = (T01 / "solution/pipeline.py").resolve()
    runs = [
        t01.grade(sub, t01_dev_seeds[:3], LocalRunner((sys.executable,)), tmp_path / str(i))
        for i in range(2)
    ]
    assert runs[0].to_json() == runs[1].to_json()


def test_case_generation_is_deterministic(t01, t01_dev_seeds) -> None:
    a, b = t01.make_case(t01_dev_seeds[0]), t01.make_case(t01_dev_seeds[0])
    assert a.raw.tobytes() == b.raw.tobytes()
    assert a.r_idx.tobytes() == b.r_idx.tobytes()
