"""python -m pkmodel.cli --params P.json --regimen R.json --out conc.csv"""

import argparse

import numpy as np

from pkmodel.grid import output_times
from pkmodel.regimen import load_regimen
from pkmodel.solver import simulate
from pkmodel.units import load_params


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pkmodel")
    ap.add_argument("--params", required=True)
    ap.add_argument("--regimen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    params = load_params(a.params)
    doses, grid = load_regimen(a.regimen)
    t = output_times(grid["start_h"], grid["stop_h"], grid["step_h"])
    c = simulate(params, doses, t)
    np.savetxt(a.out, np.column_stack([t, c]), delimiter=",", header="time_h,conc_mg_per_L",
               comments="", fmt="%.12g")


if __name__ == "__main__":
    main()
