"""Parameter input and unit conversion."""

import json
from dataclasses import dataclass

ML_PER_L = 1000.0
MIN_PER_H = 60.0


@dataclass(frozen=True)
class Params:
    cl_l_per_h: float
    v1_l: float
    q_l_per_h: float
    v2_l: float


def clearance_l_per_h(cl_ml_per_min: float) -> float:
    return cl_ml_per_min / ML_PER_L / MIN_PER_H


def load_params(path) -> Params:
    with open(path) as fh:
        raw = json.load(fh)
    p = Params(
        cl_l_per_h=clearance_l_per_h(raw["CL_mL_per_min"]),
        v1_l=raw["V1_L"],
        q_l_per_h=raw["Q_L_per_h"],
        v2_l=raw["V2_L"],
    )
    if min(p.cl_l_per_h, p.v1_l, p.q_l_per_h, p.v2_l) <= 0:
        raise ValueError("parameters must be positive")
    return p
