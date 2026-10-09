"""t02 grader validation: references pass, realistic mistakes fail with the right code."""

import json
import sys

import numpy as np
import pytest
from conftest import T02, write_t02_outputs

from bioengbench.core.runner import LocalRunner
from bioengbench.core.scoring import Status
from bioengbench.models.pk import concentrations
from bioengbench.tasks import t02_pk as T

SEED = 0x2C1B_7D40_993E_A5F1_0B62_4E18_D7C3_5A90


@pytest.fixture(scope="module")
def case(t02):
    return t02.make_case(SEED)


def _codes(t02, out, case) -> set[str]:
    return {r.code for r in t02.evaluate_outputs(out, case) if r.status is Status.FAIL}


def _regimen(out):
    return json.loads((out / "regimen.json").read_text())


def test_perfect_submission_passes(t02, case, tmp_path) -> None:
    results = t02.evaluate_outputs(write_t02_outputs(tmp_path, case), case)
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_true_parameters_are_inside_the_likelihood_region(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case, params=case.p_true)
    assert "PARAMS_LR" not in _codes(t02, out, case)


@pytest.mark.parametrize(
    ("scale", "label"),
    [((1, 1000, 1, 1000), "volumes in mL"), ((1 / 60, 1, 1 / 60, 1), "clearances per minute")],
)
def test_parameter_unit_slips(t02, case, tmp_path, scale, label) -> None:
    out = write_t02_outputs(tmp_path, case, params=case.p_mle * np.array(scale))
    assert "PARAMS_LR" in _codes(t02, out, case), label


def test_one_compartment_collapse_fails_likelihood(t02, case, tmp_path) -> None:
    p = case.p_mle.copy()
    p[2] = 1e-6  # Q -> 0: the peripheral compartment is ignored
    assert "PARAMS_LR" in _codes(t02, write_t02_outputs(tmp_path, case, params=p), case)


def test_explicit_euler_prediction_fails_solver_accuracy(t02, case, tmp_path) -> None:
    from bioengbench.models.pk import rate_matrix

    k, dt, x, pred = rate_matrix(case.p_mle), 0.25, np.zeros(2), []
    for t in T.PRED_GRID:
        pred.append(x[0] / case.p_mle[1])
        x = x + dt * (k @ x + np.array([1500.0 if t < 1.0 else 0.0, 0.0]))
    out = write_t02_outputs(tmp_path, case, pred=np.array(pred))
    assert "PRED_ACCURACY" in _codes(t02, out, case)


def test_bolus_instead_of_infusion_fails_solver_accuracy(t02, case, tmp_path) -> None:
    from bioengbench.models.pk import Infusion

    pred = concentrations(case.p_mle, [Infusion(0.0, 1e-6, 1500.0)], T.PRED_GRID)
    out = write_t02_outputs(tmp_path, case, pred=pred)
    assert "PRED_ACCURACY" in _codes(t02, out, case)


def test_standard_errors_on_linear_scale_fail(t02, case, tmp_path) -> None:
    se_linear = T.fisher_se(case.p_mle) * case.p_mle
    out = write_t02_outputs(tmp_path, case, se=se_linear)
    assert "SE_CONSISTENT" in _codes(t02, out, case)


def test_half_life_without_ln2_fails(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case)
    doc = json.loads((out / "params.json").read_text())
    doc["t_half_beta_h"] /= np.log(2)
    (out / "params.json").write_text(json.dumps(doc))
    assert "HALF_LIVES" in _codes(t02, out, case)


def test_daily_dose_reported_as_per_interval_dose_is_unsafe(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case)
    reg = _regimen(out)
    reg["dose_mg"] *= 24 / reg["tau_h"]
    (out / "regimen.json").write_text(json.dumps(reg))
    assert {"REGIMEN_SAFE", "REGIMEN_EFFICIENT", "REGIMEN_SELF"} <= _codes(t02, out, case)


def test_disallowed_interval_fails_format(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case)
    reg = _regimen(out)
    reg["tau_h"] = 10.0
    (out / "regimen.json").write_text(json.dumps(reg))
    codes = _codes(t02, out, case)
    assert "REGIMEN_FORMAT" in codes and "REGIMEN_SAFE" not in codes  # gated, skipped


def test_swapped_peak_and_trough_fail(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case)
    reg = _regimen(out)
    reg["predicted_peak_mg_per_L"], reg["predicted_trough_mg_per_L"] = (
        reg["predicted_trough_mg_per_L"],
        reg["predicted_peak_mg_per_L"],
    )
    (out / "regimen.json").write_text(json.dumps(reg))
    assert "REGIMEN_SELF" in _codes(t02, out, case)


def test_wrong_prediction_grid_fails(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case)
    grid = np.arange(0, 48.01, 0.5)
    c = concentrations(case.p_mle, T.LOADING, grid)
    np.savetxt(
        out / "prediction.csv",
        np.column_stack([grid, c]),
        delimiter=",",
        header="time_h,conc_mg_per_L",
        comments="",
        fmt="%.10g",
    )
    assert "PRED_GRID" in _codes(t02, out, case)


def test_negative_parameter_fails_schema(t02, case, tmp_path) -> None:
    p = case.p_mle.copy()
    p[3] = -p[3]
    out = write_t02_outputs(tmp_path, case, params=np.abs(p), se=T.fisher_se(case.p_mle))
    doc = json.loads((out / "params.json").read_text())
    doc["V2_L"] = -1.0
    (out / "params.json").write_text(json.dumps(doc))
    assert "SCHEMA" in _codes(t02, out, case)


@pytest.mark.slow
def test_python_reference_passes_dev_seeds(t02, t02_dev_seeds, tmp_path) -> None:
    sub = (T02 / "solution/pk_pipeline.py").resolve()
    verdict = t02.grade(sub, t02_dev_seeds, LocalRunner((sys.executable,)), tmp_path)
    assert verdict.passed, verdict.to_json()


@pytest.mark.slow
@pytest.mark.octave
def test_octave_reference_passes_dev_seeds(t02, t02_dev_seeds, octave, tmp_path) -> None:
    sub = (T02 / "references/pk_pipeline.m").resolve()
    runner = LocalRunner((octave, "--no-gui", "--quiet"))
    verdict = t02.grade(sub, t02_dev_seeds[:4], runner, tmp_path, "octave")
    assert verdict.passed, verdict.to_json()


def test_designing_to_the_window_edges_fails_targets(t02, case, tmp_path) -> None:
    """Designed to [10, 40] instead of the targets: may still be safe at the
    truth (27 of 40 patients in the campaign), but it ignores the margin."""
    from bioengbench.models.pk import optimal_regimen, steady_state

    reg = optimal_regimen(case.p_mle, T.WINDOW, T.TAUS, T.INFUSION_H)
    ss = steady_state(case.p_mle, reg.dose_mg, reg.tau_h, T.INFUSION_H)
    regimen = {
        "tau_h": reg.tau_h,
        "dose_mg": reg.dose_mg,
        "infusion_h": T.INFUSION_H,
        "predicted_peak_mg_per_L": ss.peak_mg_per_l,
        "predicted_trough_mg_per_L": ss.trough_mg_per_l,
    }
    out = write_t02_outputs(tmp_path, case)
    (out / "regimen.json").write_text(json.dumps(regimen))
    codes = _codes(t02, out, case)
    assert "REGIMEN_TARGETS" in codes
    assert "REGIMEN_SELF" not in codes
