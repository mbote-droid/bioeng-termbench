"""Shared fixtures for task-level tests."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from bioengbench.core import registry
from bioengbench.core.seeds import read_seed_file
from bioengbench.tasks.t01_ecg import hrv

TASKS = Path(__file__).resolve().parents[2] / "tasks"
T01 = TASKS / "t01-ecg-denoise-rpeaks"


@pytest.fixture(scope="session")
def t01():
    return registry.get("t01-ecg-denoise-rpeaks")


@pytest.fixture(scope="session")
def t01_dev_seeds() -> list[int]:
    return read_seed_file(T01 / "tests" / "dev_seeds.txt")


@pytest.fixture(scope="session")
def octave() -> str:
    exe = shutil.which("octave-cli")
    if exe is None:
        pytest.skip("Octave not installed")
    return exe


def write_t01_outputs(out: Path, case, filtered=None, peaks=None, metrics=None) -> Path:
    """A submission built from ground truth, optionally with one part replaced."""
    out.mkdir(parents=True, exist_ok=True)
    y = case.clean if filtered is None else filtered
    p = case.r_idx if peaks is None else np.asarray(peaks)
    m = hrv(case.r_idx, case.fs) if metrics is None else metrics
    np.savetxt(
        out / "ecg_filtered.csv",
        np.column_stack([case.t, y]),
        delimiter=",",
        header="time_s,ecg_mV",
        comments="",
        fmt="%.6f",
    )
    np.savetxt(
        out / "rpeaks.csv",
        np.column_stack([p, p / case.fs]),
        delimiter=",",
        header="sample_index,time_s",
        comments="",
        fmt=["%d", "%.6f"],
    )
    (out / "metrics.json").write_text(json.dumps(m))
    return out


T02 = TASKS / "t02-pk-fit-dose"


@pytest.fixture(scope="session")
def t02():
    return registry.get("t02-pk-fit-dose")


@pytest.fixture(scope="session")
def t02_dev_seeds() -> list[int]:
    return read_seed_file(T02 / "tests" / "dev_seeds.txt")


def write_t02_outputs(out: Path, case, params=None, se=None, pred=None, regimen=None) -> Path:
    """A correct submission built from the grader's own MLE, optionally altered."""
    from bioengbench.models.pk import concentrations, half_lives, optimal_regimen, steady_state
    from bioengbench.tasks import t02_pk as T

    out.mkdir(parents=True, exist_ok=True)
    p = case.p_mle if params is None else np.asarray(params)
    se = T.fisher_se(p) if se is None else se
    alpha, beta = half_lives(p)
    doc = {
        "CL_L_per_h": p[0],
        "V1_L": p[1],
        "Q_L_per_h": p[2],
        "V2_L": p[3],
        "se_log": dict(zip(["CL", "V1", "Q", "V2"], map(float, se))),
        "t_half_alpha_h": alpha,
        "t_half_beta_h": beta,
    }
    (out / "params.json").write_text(json.dumps(doc))
    grid = T.PRED_GRID
    c = concentrations(p, T.LOADING, grid) if pred is None else pred
    np.savetxt(
        out / "prediction.csv",
        np.column_stack([grid, c]),
        delimiter=",",
        header="time_h,conc_mg_per_L",
        comments="",
        fmt="%.10g",
    )
    if regimen is None:
        reg = optimal_regimen(p, T.TARGET, T.TAUS, T.INFUSION_H)
        ss = steady_state(p, reg.dose_mg, reg.tau_h, T.INFUSION_H)
        regimen = {
            "tau_h": reg.tau_h,
            "dose_mg": reg.dose_mg,
            "infusion_h": 1.0,
            "predicted_peak_mg_per_L": ss.peak_mg_per_l,
            "predicted_trough_mg_per_L": ss.trough_mg_per_l,
        }
    (out / "regimen.json").write_text(json.dumps(regimen))
    return out


T03 = TASKS / "t03-glucose-closed-loop"


@pytest.fixture(scope="session")
def t03():
    return registry.get("t03-glucose-closed-loop")


@pytest.fixture(scope="session")
def t03_dev_seeds() -> list[int]:
    return read_seed_file(T03 / "tests" / "dev_seeds.txt")


T04 = TASKS / "t04-bioheat-ablation"


@pytest.fixture(scope="session")
def t04():
    return registry.get("t04-bioheat-ablation")


@pytest.fixture(scope="session")
def t04_dev_seeds() -> list[int]:
    return read_seed_file(T04 / "tests" / "dev_seeds.txt")


T05 = TASKS / "t05-pk-debug"


@pytest.fixture(scope="session")
def t05():
    return registry.get("t05-pk-debug")


@pytest.fixture(scope="session")
def t05_dev_seeds() -> list[int]:
    return read_seed_file(T05 / "tests" / "dev_seeds.txt")
