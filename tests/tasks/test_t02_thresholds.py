"""t02 acceptance thresholds pinned at their boundaries, with the values the
contract states (instruction.md), not the grader's constants, so a changed
constant fails here. Added after grader mutation testing: halving or
doubling most t02 tolerances went unnoticed, and the FILES and SCHEMA gates
could be switched off because later checks also fail on missing files."""

import json

import numpy as np
import pytest
from conftest import write_t02_outputs
from scipy.optimize import brentq
from scipy.stats import chi2

from bioengbench.core.scoring import Status
from bioengbench.models.pk import concentrations, half_lives, steady_state
from bioengbench.tasks import t02_pk as T

SEED = 0x2C1B_7D40_993E_A5F1_0B62_4E18_D7C3_5A90
SIGMA = 0.10  # contract: known assay error
LR = chi2.ppf(0.999, 4)  # contract: 99.9 % likelihood-ratio region


@pytest.fixture(scope="module")
def case(t02):
    return t02.make_case(SEED)


def _failed(t02, out, case) -> set[str]:
    return {r.code for r in t02.evaluate_outputs(out, case) if r.status is Status.FAIL}


def _status(t02, out, case, code) -> Status:
    return next(r.status for r in t02.evaluate_outputs(out, case) if r.code == code)


def test_contract_constants() -> None:
    assert pytest.approx(18.47, abs=0.01) == LR


@pytest.mark.parametrize(("side", "expected"), [(0.98, Status.PASS), (1.02, Status.FAIL)])
def test_likelihood_region_boundary(t02, case, tmp_path, side, expected) -> None:
    """Move CL away from the MLE until the LR statistic (with the contract's
    sigma) sits just inside or just outside the region."""

    def stat(k: float) -> float:
        p = case.p_mle * np.array([np.exp(k), 1, 1, 1])
        return (T.ssr(p, case.obs) - case.ssr_mle) / SIGMA**2

    k = brentq(lambda k: stat(k) - side * LR, 0.0, 2.0, xtol=1e-12)
    p = case.p_mle * np.array([np.exp(k), 1, 1, 1])
    out = write_t02_outputs(tmp_path, case, params=p)
    assert _status(t02, out, case, "PARAMS_LR") is expected


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(1.19, Status.PASS), (1.21, Status.FAIL), (0.81, Status.PASS), (0.79, Status.FAIL)],
)
def test_standard_error_boundary(t02, case, tmp_path, factor, expected) -> None:
    se = T.fisher_se(case.p_mle)
    se[0] *= factor
    out = write_t02_outputs(tmp_path, case, se=se)
    assert _status(t02, out, case, "SE_CONSISTENT") is expected


def test_unidentified_parameter_may_be_reported_as_poorly_determined(t02, case, tmp_path) -> None:
    """With Q ten times smaller, the Fisher SE of log Q is about 1.08: not
    identified. Any SE >= 0.5 is accepted for it; a precise-looking one is not."""
    p = case.p_mle * np.array([1, 1, 0.1, 1])
    se = T.fisher_se(p)
    assert 1.0 < se[2] < 2.0
    se[2] = 0.6
    out = write_t02_outputs(tmp_path / "a", case, params=p, se=se)
    assert _status(t02, out, case, "SE_CONSISTENT") is Status.PASS
    se[2] = 0.3
    out = write_t02_outputs(tmp_path / "b", case, params=p, se=se)
    assert _status(t02, out, case, "SE_CONSISTENT") is Status.FAIL


def test_borderline_identified_parameter_is_compared_relatively(t02, case, tmp_path) -> None:
    """With Q five times smaller the SE of log V2 is about 0.7: identified, so
    it must match within 20 %."""
    p = case.p_mle * np.array([1, 1, 0.2, 1])
    se = T.fisher_se(p)
    assert 0.5 < se[3] < 1.0
    se[3] *= 0.45
    out = write_t02_outputs(tmp_path, case, params=p, se=se)
    assert _status(t02, out, case, "SE_CONSISTENT") is Status.FAIL


@pytest.mark.parametrize(("factor", "expected"), [(1.0049, Status.PASS), (1.0051, Status.FAIL)])
def test_half_life_boundary(t02, case, tmp_path, factor, expected) -> None:
    out = write_t02_outputs(tmp_path, case)
    doc = json.loads((out / "params.json").read_text())
    alpha, _ = half_lives(case.p_mle)
    doc["t_half_alpha_h"] = alpha * factor
    (out / "params.json").write_text(json.dumps(doc))
    assert _status(t02, out, case, "HALF_LIVES") is expected


@pytest.mark.parametrize(("frac", "expected"), [(0.0099, Status.PASS), (0.0101, Status.FAIL)])
def test_prediction_accuracy_boundary(t02, case, tmp_path, frac, expected) -> None:
    exact = concentrations(case.p_mle, T.LOADING, T.PRED_GRID)
    pred = exact.copy()
    pred[40] += frac * exact.max()
    out = write_t02_outputs(tmp_path, case, pred=pred)
    assert _status(t02, out, case, "PRED_ACCURACY") is expected


def _regimen(case, dose_factor=1.0, peak_factor=1.0):
    p = case.p_mle
    from bioengbench.models.pk import optimal_regimen

    own = optimal_regimen(p, T.TARGET, T.TAUS, T.INFUSION_H)
    dose = own.dose_mg * dose_factor
    ss = steady_state(p, dose, own.tau_h, T.INFUSION_H)
    return {
        "tau_h": own.tau_h,
        "dose_mg": dose,
        "infusion_h": 1.0,
        "predicted_peak_mg_per_L": ss.peak_mg_per_l * peak_factor,
        "predicted_trough_mg_per_L": ss.trough_mg_per_l,
    }


@pytest.mark.parametrize(("factor", "expected"), [(1.0099, Status.PASS), (1.0101, Status.FAIL)])
def test_steady_state_self_consistency_boundary(t02, case, tmp_path, factor, expected) -> None:
    out = write_t02_outputs(tmp_path, case, regimen=_regimen(case, peak_factor=factor))
    assert _status(t02, out, case, "REGIMEN_SELF") is expected


@pytest.mark.parametrize(("ratio", "expected"), [(1.19, Status.PASS), (1.21, Status.FAIL)])
def test_efficiency_boundary(t02, case, tmp_path, ratio, expected) -> None:
    """Daily dose at 1.19 / 1.21 x the true optimum (contract: within 20 %)."""
    opt = case.optimum
    dose = opt.dose_mg * ratio
    ss = steady_state(case.p_mle, dose, opt.tau_h, T.INFUSION_H)
    reg = {
        "tau_h": opt.tau_h,
        "dose_mg": dose,
        "infusion_h": 1.0,
        "predicted_peak_mg_per_L": ss.peak_mg_per_l,
        "predicted_trough_mg_per_L": ss.trough_mg_per_l,
    }
    out = write_t02_outputs(tmp_path, case, regimen=reg)
    assert _status(t02, out, case, "REGIMEN_EFFICIENT") is expected


def test_missing_file_fails_the_files_gate(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case)
    (out / "regimen.json").unlink()
    assert "FILES" in _failed(t02, out, case)


def test_malformed_file_fails_the_schema_gate(t02, case, tmp_path) -> None:
    out = write_t02_outputs(tmp_path, case)
    doc = json.loads((out / "params.json").read_text())
    doc["V1_L"] = "twenty"
    (out / "params.json").write_text(json.dumps(doc))
    assert "SCHEMA" in _failed(t02, out, case)


def _regimen_at(case, tau, dose):
    ss = steady_state(case.p_mle, dose, tau, 1.0)
    return {
        "tau_h": tau,
        "dose_mg": dose,
        "infusion_h": 1.0,
        "predicted_peak_mg_per_L": ss.peak_mg_per_l,
        "predicted_trough_mg_per_L": ss.trough_mg_per_l,
    }


@pytest.mark.parametrize(("factor", "expected"), [(0.991, Status.PASS), (0.989, Status.FAIL)])
def test_design_target_trough_boundary(t02, case, tmp_path, factor, expected) -> None:
    """Contract: predicted trough >= 11.5 mg/L, 1 % tolerance. Uses the
    interval where the targets are feasible, so only the trough is at issue."""
    from bioengbench.models.pk import optimal_regimen

    tau = optimal_regimen(case.p_mle, (11.5, 35.0), (6.0, 8.0, 12.0, 24.0), 1.0).tau_h
    per_mg = steady_state(case.p_mle, 1.0, tau, 1.0).trough_mg_per_l
    reg = _regimen_at(case, tau, 11.5 * factor / per_mg)
    out = write_t02_outputs(tmp_path, case, regimen=reg)
    assert _status(t02, out, case, "REGIMEN_TARGETS") is expected


@pytest.mark.parametrize(("factor", "expected"), [(1.009, Status.PASS), (1.011, Status.FAIL)])
def test_design_target_peak_boundary(t02, case, tmp_path, factor, expected) -> None:
    """Contract: predicted peak <= 35 mg/L, 1 % tolerance (6 h interval so the
    trough stays above its target at this dose)."""
    tau = 6.0
    per_mg = steady_state(case.p_mle, 1.0, tau, 1.0)
    reg = _regimen_at(case, tau, 35.0 * factor / per_mg.peak_mg_per_l)
    assert reg["predicted_trough_mg_per_L"] >= 11.5
    out = write_t02_outputs(tmp_path, case, regimen=reg)
    assert _status(t02, out, case, "REGIMEN_TARGETS") is expected
