"""t05 tolerances pinned at their boundaries (found necessary by grader
mutation testing: every defect produces errors far beyond the tolerances, so
the end-to-end tests could not notice a tolerance being halved or doubled)."""

import numpy as np
import pytest

from bioengbench.checks.base import CheckContext
from bioengbench.core.scoring import Status
from bioengbench.tasks import t05_pk_debug as T

SEED = 0x51E5_7A0B_12C4_9D3E_8F60_A1B2_C3D4_E5F6


def _ctx(tmp_path, name, dt=0.0, dc_rel=0.0):
    case = T.make_case(SEED)
    scn = next(s for s in case.scenarios if s.name == name)
    t = scn.times()
    ref = T.exact_concentrations(scn.params, scn.doses, t)
    out = tmp_path / name
    out.mkdir()
    c = ref + dc_rel * ref.max()
    np.savetxt(
        out / "conc.csv",
        np.column_stack([t + dt, c]),
        delimiter=",",
        header="time_h,conc_mg_per_L",
        comments="",
        fmt="%.17g",
    )
    return CheckContext(output_dir=tmp_path, truth=case, run_dirs={name: out})


def _status(ctx, name, code):
    return T.scenario_check(name, code).evaluate(ctx)[0].status


@pytest.mark.parametrize(("dc", "expected"), [(0.9e-5, Status.PASS), (1.1e-5, Status.FAIL)])
def test_concentration_tolerance_boundary(tmp_path, dc, expected) -> None:
    assert _status(_ctx(tmp_path, "standard", dc_rel=dc), "standard", "STANDARD") is expected


@pytest.mark.parametrize(("dt", "expected"), [(0.9e-9, Status.PASS), (1.1e-9, Status.FAIL)])
def test_time_grid_tolerance_boundary(tmp_path, dt, expected) -> None:
    assert (
        _status(_ctx(tmp_path, "long_horizon", dt=dt), "long_horizon", "LONG_HORIZON") is expected
    )


def test_exact_output_passes_every_scenario(tmp_path) -> None:
    for name, code in T.SCENARIO_CODES:
        assert _status(_ctx(tmp_path, name), name, code) is Status.PASS
