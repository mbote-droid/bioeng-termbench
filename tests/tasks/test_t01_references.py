"""False-reject side of grader validity: correct solutions must pass on every dev seed.

Three independent correct implementations:
  oracle (Python): IIR Butterworth + notch, Pan-Tompkins detector
  rbj (Octave):    hand-built biquads, same algorithm family, different language
  spline (Python): least-squares mains fit, spline TP baseline, Hilbert detector
One boundary probe (median baseline) must fail on FIDELITY only.
"""

import sys

import pytest
from conftest import T01

from bioengbench.core.runner import LocalRunner


def _grade(task, sub, seeds, runner, tmp_path, lang="python"):
    return task.grade(sub.resolve(), seeds, runner, tmp_path, lang)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["solution/pipeline.py", "references/pipeline_spline_hilbert.py"])
def test_python_references_pass(name, t01, t01_dev_seeds, tmp_path) -> None:
    verdict = _grade(t01, T01 / name, t01_dev_seeds, LocalRunner((sys.executable,)), tmp_path)
    assert verdict.passed, verdict.to_json()


@pytest.mark.slow
@pytest.mark.octave
def test_octave_reference_passes(t01, t01_dev_seeds, octave, tmp_path) -> None:
    runner = LocalRunner((octave, "--no-gui", "--quiet"))
    sub = T01 / "references/pipeline_rbj.m"
    verdict = _grade(t01, sub, t01_dev_seeds, runner, tmp_path, "octave")
    assert verdict.passed, verdict.to_json()


@pytest.mark.slow
def test_median_baseline_probe_fails_on_fidelity_only(t01, t01_dev_seeds, tmp_path) -> None:
    sub = T01 / "references/pipeline_median_hilbert.py"
    verdict = _grade(t01, sub, t01_dev_seeds, LocalRunner((sys.executable,)), tmp_path)
    assert not verdict.passed
    assert verdict.failure_codes() == {"FIDELITY"}
