"""t02: fit a two-compartment PK model to sparse noisy data and design a dosing regimen.

The agent gets one patient's plasma concentrations after a 1 h loading
infusion, fits CL, V1, Q and V2, reports their uncertainty, predicts the
concentration curve with its own solver, and designs a maintenance regimen
that keeps steady state inside a therapeutic window.

How each output is judged, and why:

* Parameters are judged by the *data*, not by the hidden truth. With 14 noisy
  samples some parameters are only weakly identifiable (for example V1 when
  distribution is fast), so the maximum-likelihood estimate can legitimately
  sit far from the truth. A truth-based tolerance would then punish a correct
  fit. Instead the estimate must lie inside the 99.9 % likelihood-ratio
  confidence region: (SSR_agent - SSR_mle) / sigma^2 <= chi2_4(0.999) = 18.47.
  On 150 simulated patients the true parameters themselves scored at most 15.5.
* Standard errors must match the Fisher information at the agent's own
  estimate (known sigma), within 20 %.
* The predicted curve must match the exact (matrix-exponential) solution of
  the agent's own parameters to 1 % of peak, a tenth of the 10 % assay noise.
  This isolates solver accuracy from fitting accuracy.
* The regimen is judged against the hidden truth: steady state must stay in
  the full window [10, 40] mg/L, and the daily dose must be within 20 % of the
  true optimum for the design targets [11.5, 35]. The gap between targets and
  window is the safety margin that absorbs estimation error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from bioengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from bioengbench.checks.io import FloatArray, SchemaError, read_csv_columns, read_json_numbers
from bioengbench.core.registry import register
from bioengbench.core.seeds import streams
from bioengbench.core.task import TaskPlugin
from bioengbench.models.pk import (
    Infusion,
    Regimen,
    concentrations,
    half_lives,
    optimal_regimen,
    steady_state,
)

T_OBS = np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0])
LOADING = [Infusion(0.0, 1.0, 1500.0)]
SIGMA_LOG = 0.10  # log-normal proportional assay error, known
PARAM_LO = np.log([3.0, 15.0, 5.0, 20.0])
PARAM_HI = np.log([7.0, 30.0, 15.0, 50.0])
WINDOW = (10.0, 40.0)  # mg/L, therapeutic window (judged against truth)
TARGET = (11.5, 35.0)  # mg/L, design targets the agent must aim for
TAUS = (6.0, 8.0, 12.0, 24.0)
INFUSION_H = 1.0
PRED_GRID = np.round(np.arange(0.0, 48.0 + 1e-9, 0.25), 10)

PARAM_KEYS = ("CL_L_per_h", "V1_L", "Q_L_per_h", "V2_L", "t_half_alpha_h", "t_half_beta_h")
SE_KEYS = ("CL", "V1", "Q", "V2")
REGIMEN_KEYS = (
    "tau_h",
    "dose_mg",
    "infusion_h",
    "predicted_peak_mg_per_L",
    "predicted_trough_mg_per_L",
)

LR_LIMIT = 18.47  # chi-square, 4 dof, 0.999 quantile
SE_REL_TOL = 0.20
SE_UNIDENTIFIED = 1.0  # grader SE above this (log units) means "unidentified"
HALF_LIFE_REL_TOL = 0.005
PRED_REL_TOL = 0.01  # of peak concentration
SS_REL_TOL = 0.01
EFFICIENCY_LIMIT = 1.20  # doses are rounded to vial sizes (~10 %) in practice
FEASIBILITY_MARGIN = 0.85  # truth must allow d_min <= 0.85 d_max for some tau


@dataclass(frozen=True)
class PkCase:
    p_true: FloatArray
    obs: FloatArray
    p_mle: FloatArray
    ssr_mle: float
    optimum: Regimen


def log_residuals(log_p: FloatArray, obs: FloatArray) -> FloatArray:
    pred = concentrations(np.exp(log_p), LOADING, T_OBS)
    out: FloatArray = np.log(np.maximum(pred, 1e-300)) - np.log(obs)
    return out


def ssr(p: FloatArray, obs: FloatArray) -> float:
    return float(np.sum(log_residuals(np.log(p), obs) ** 2))


def fisher_se(p: FloatArray) -> FloatArray:
    """SE of log-parameters from the Fisher information with known sigma.

    J = d log C / d log p (central differences); F = J^T J / sigma^2.
    """
    lp = np.log(p)
    h = 1e-5
    cols = []
    for i in range(4):
        step = np.zeros(4)
        step[i] = h
        up = np.log(concentrations(np.exp(lp + step), LOADING, T_OBS))
        dn = np.log(concentrations(np.exp(lp - step), LOADING, T_OBS))
        cols.append((up - dn) / (2 * h))
    j = np.column_stack(cols)
    cov = np.linalg.pinv(j.T @ j / SIGMA_LOG**2)
    se: FloatArray = np.sqrt(np.abs(np.diag(cov)))
    return se


def fit_mle(obs: FloatArray, rng: np.random.Generator) -> tuple[FloatArray, float]:
    """Multi-start bounded Levenberg-Marquardt-type fit in log-parameter space."""
    starts = [np.log([4.5, 20.0, 9.0, 32.0])]
    starts += [rng.uniform(np.log([2, 8, 2, 8]), np.log([12, 60, 40, 120])) for _ in range(8)]
    bounds = (np.log([0.1, 1.0, 0.1, 1.0]), np.log([100.0, 500.0, 500.0, 2000.0]))
    best = None
    for start in starts:
        r = least_squares(log_residuals, start, args=(obs,), method="trf", bounds=bounds)
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    p = np.exp(best.x)
    return p, ssr(p, obs)


def _feasible_with_margin(p: FloatArray) -> bool:
    for tau in TAUS:
        unit = steady_state(p, 1.0, tau, INFUSION_H)
        if TARGET[0] / unit.trough_mg_per_l <= FEASIBILITY_MARGIN * TARGET[1] / unit.peak_mg_per_l:
            return True
    return False


@lru_cache(maxsize=64)
def make_case(seed: int) -> PkCase:
    rng_patient, rng_noise, rng_fit = streams(seed, 3)
    # Rejection sampling: only patients for whom the target window is reachable
    # with margin. Patients outside it need continuous infusion, not this task.
    while True:
        p = np.exp(rng_patient.uniform(PARAM_LO, PARAM_HI))
        if _feasible_with_margin(p):
            break
    obs = concentrations(p, LOADING, T_OBS) * np.exp(rng_noise.normal(0.0, SIGMA_LOG, T_OBS.size))
    p_mle, ssr_mle = fit_mle(obs, rng_fit)
    optimum = optimal_regimen(p, TARGET, TAUS, INFUSION_H)
    assert optimum is not None
    return PkCase(p, obs, p_mle, ssr_mle, optimum)


def write_inputs(case: PkCase, input_dir: Path) -> None:
    np.savetxt(
        input_dir / "observations.csv",
        np.column_stack([T_OBS, case.obs]),
        delimiter=",",
        header="time_h,conc_mg_per_L",
        comments="",
        fmt="%.6g",
    )
    problem = {
        "doses": [
            {"start_h": d.start_h, "duration_h": d.duration_h, "amount_mg": d.amount_mg}
            for d in LOADING
        ],
        "error_model": {"type": "lognormal_proportional", "sigma_log": SIGMA_LOG},
        "therapeutic_window_mg_per_L": list(WINDOW),
        "design_targets_mg_per_L": list(TARGET),
        "allowed_tau_h": list(TAUS),
        "maintenance_infusion_h": INFUSION_H,
        "prediction_grid_h": {"start": 0.0, "stop": 48.0, "step": 0.25},
    }
    (input_dir / "problem.json").write_text(json.dumps(problem, indent=2) + "\n")


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> PkCase:
    case: PkCase = ctx.truth
    return case


def _params(ctx: CheckContext) -> FloatArray:
    m = ctx.outputs["params"]
    return np.array([m["CL_L_per_h"], m["V1_L"], m["Q_L_per_h"], m["V2_L"]])


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    names = ("params.json", "prediction.csv", "regimen.json")
    missing = [n for n in names if not (ctx.output_dir / n).is_file()]
    return Outcome(not missing, f"missing: {', '.join(missing)}" if missing else "")


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    params = read_json_numbers(ctx.output_dir / "params.json", PARAM_KEYS)
    raw = json.loads((ctx.output_dir / "params.json").read_text())
    se = raw.get("se_log")
    if not isinstance(se, dict):
        raise SchemaError("params.json: 'se_log' must be an object")
    for key in SE_KEYS:
        value = se.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise SchemaError(f"params.json: se_log.{key} must be a positive number")
    p = np.array([params[k] for k in PARAM_KEYS[:4]])
    if not np.all(p > 0):
        raise SchemaError("parameters must be positive")
    ctx.outputs["params"] = params
    ctx.outputs["se_log"] = np.array([float(se[k]) for k in SE_KEYS])
    ctx.outputs["pred"] = read_csv_columns(
        ctx.output_dir / "prediction.csv", ("time_h", "conc_mg_per_L")
    )
    ctx.outputs["regimen"] = read_json_numbers(ctx.output_dir / "regimen.json", REGIMEN_KEYS)
    return Outcome(True)


@check("PARAMS_LR")
def params_likelihood(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    stat = (ssr(_params(ctx), case.obs) - case.ssr_mle) / SIGMA_LOG**2
    msg = "parameters outside the 99.9 % likelihood region"
    return Outcome(stat <= LR_LIMIT, msg, stat, LR_LIMIT)


@check("SE_CONSISTENT")
def se_consistent(ctx: CheckContext) -> Outcome:
    ref = fisher_se(_params(ctx))
    got = ctx.outputs["se_log"]
    worst = 0.0
    for r, g in zip(ref, got, strict=True):
        if r > SE_UNIDENTIFIED:
            if g < 0.5 * SE_UNIDENTIFIED:  # must at least report it as poorly determined
                return Outcome(False, "an unidentified parameter was reported as precise")
            continue
        worst = max(worst, abs(g / r - 1.0))
    return Outcome(
        worst <= SE_REL_TOL,
        "standard errors do not match the Fisher information",
        worst,
        SE_REL_TOL,
    )


@check("HALF_LIVES")
def half_life_check(ctx: CheckContext) -> Outcome:
    a, b = half_lives(_params(ctx))
    m = ctx.outputs["params"]
    worst = max(abs(m["t_half_alpha_h"] / a - 1), abs(m["t_half_beta_h"] / b - 1))
    return Outcome(
        worst <= HALF_LIFE_REL_TOL,
        "half-lives inconsistent with the parameters",
        worst,
        HALF_LIFE_REL_TOL,
    )


@check("PRED_GRID", gate=True)
def prediction_grid(ctx: CheckContext) -> Outcome:
    t = ctx.outputs["pred"]["time_h"]
    ok = t.size == PRED_GRID.size and bool(np.allclose(t, PRED_GRID, atol=1e-6))
    return Outcome(ok, "prediction.csv must use the grid 0, 0.25, ..., 48 h")


@check("PRED_ACCURACY")
def prediction_accuracy(ctx: CheckContext) -> Outcome:
    exact = concentrations(_params(ctx), LOADING, PRED_GRID)
    err = float(np.max(np.abs(ctx.outputs["pred"]["conc_mg_per_L"] - exact)) / exact.max())
    return Outcome(err <= PRED_REL_TOL, "solver error above 1 % of peak", err, PRED_REL_TOL)


@check("REGIMEN_FORMAT", gate=True)
def regimen_format(ctx: CheckContext) -> Outcome:
    r = ctx.outputs["regimen"]
    if r["tau_h"] not in TAUS:
        return Outcome(False, f"tau_h must be one of {TAUS}")
    if abs(r["infusion_h"] - INFUSION_H) > 1e-9:
        return Outcome(False, "infusion_h must be 1")
    return Outcome(r["dose_mg"] > 0, "dose_mg must be positive")


@check("REGIMEN_SELF")
def regimen_self(ctx: CheckContext) -> Outcome:
    r = ctx.outputs["regimen"]
    own = steady_state(_params(ctx), r["dose_mg"], r["tau_h"], INFUSION_H)
    worst = max(
        abs(r["predicted_peak_mg_per_L"] / own.peak_mg_per_l - 1),
        abs(r["predicted_trough_mg_per_L"] / own.trough_mg_per_l - 1),
    )
    return Outcome(
        worst <= SS_REL_TOL,
        "steady-state prediction inconsistent with own model",
        worst,
        SS_REL_TOL,
    )


@check("REGIMEN_TARGETS")
def regimen_targets(ctx: CheckContext) -> Outcome:
    """The regimen must meet the design targets in the agent's own model.

    Added after the validation campaign: a regimen designed to the window
    edges instead of the targets passed REGIMEN_SAFE on 27 of 40 patients,
    because the safety check at the truth only fails when the estimation
    error happens to point the wrong way. The targets are the stated margin
    against that error, so not designing to them is itself the mistake."""
    r = ctx.outputs["regimen"]
    lo = TARGET[0] * (1 - SS_REL_TOL)
    hi = TARGET[1] * (1 + SS_REL_TOL)
    ok = r["predicted_trough_mg_per_L"] >= lo and r["predicted_peak_mg_per_L"] <= hi
    msg = (
        f"predicted steady state {r['predicted_trough_mg_per_L']:.2f}-"
        f"{r['predicted_peak_mg_per_L']:.2f} mg/L outside the targets {TARGET}"
    )
    return Outcome(ok, msg)


@check("REGIMEN_SAFE")
def regimen_safe(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    r = ctx.outputs["regimen"]
    true_ss = steady_state(case.p_true, r["dose_mg"], r["tau_h"], INFUSION_H)
    ok = WINDOW[0] <= true_ss.trough_mg_per_l and true_ss.peak_mg_per_l <= WINDOW[1]
    msg = f"true steady state {true_ss.trough_mg_per_l:.1f}-{true_ss.peak_mg_per_l:.1f} mg/L"
    return Outcome(ok, msg + f" outside {WINDOW}")


@check("REGIMEN_EFFICIENT")
def regimen_efficient(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    r = ctx.outputs["regimen"]
    ratio = r["dose_mg"] * 24.0 / r["tau_h"] / case.optimum.daily_mg
    return Outcome(
        ratio <= EFFICIENCY_LIMIT, "daily dose well above the optimum", ratio, EFFICIENCY_LIMIT
    )


@register
class PkTask(TaskPlugin):
    task_id = "t02-pk-fit-dose"
    version = "1.0.0"
    timeout_s = 120.0  # an Octave Nelder-Mead fit takes about 15 s per case

    def languages(self) -> tuple[str, ...]:
        # The Julia reference is validated through `bioeng check-outputs` in CI.
        return ("python", "octave")

    def make_case(self, seed: int) -> PkCase:
        return make_case(seed)

    def write_inputs(self, case: PkCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        if lang == "python":
            return ("{submission}", "--indir", "{input_dir}", "--outdir", "{output_dir}")
        return ("{submission}", "{input_dir}", "{output_dir}")

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t02",
            [
                CheckGroup("load", [files_present, schema, prediction_grid], gate=True),
                CheckGroup("fit", [params_likelihood, se_consistent, half_life_check]),
                CheckGroup("solver", [prediction_accuracy]),
                CheckGroup(
                    "regimen",
                    [
                        regimen_format,
                        regimen_self,
                        regimen_targets,
                        regimen_safe,
                        regimen_efficient,
                    ],
                ),
            ],
        )
