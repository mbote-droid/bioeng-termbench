"""Proportional controller with 5x the reference gain and no safety layers:
fast correction, but overshoots into hypoglycaemia."""

import json
import sys


def serve(policy):
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        sys.stdout.write(json.dumps({"insulin_mU_per_min": policy(msg, state)}) + "\n")
        sys.stdout.flush()


def policy(m, s):
    g = m["cgm_mg_dL"] if m["cgm_mg_dL"] is not None else s.get("g", 115.0)
    s["g"] = g
    b = m["basal_mU_per_min"]
    return max(0.0, min(m["pump_max_mU_per_min"], b + b / 10.0 * (g - 115.0)))


serve(policy)
