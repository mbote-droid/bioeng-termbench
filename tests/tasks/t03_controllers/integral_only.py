"""Pure integral control: slow phase lag gives a sustained oscillation."""

import json
import sys


def serve(policy):
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        sys.stdout.write(json.dumps({"insulin_mU_per_min": policy(msg, state)}) + "\n")
        sys.stdout.flush()


def policy(m, s):
    b = m["basal_mU_per_min"]
    g = m["cgm_mg_dL"] if m["cgm_mg_dL"] is not None else s.get("g", 115.0)
    s["g"] = g
    s["i"] = s.get("i", 0.0) + b / 400.0 * (g - 115.0) * 5.0
    return max(0.0, min(m["pump_max_mU_per_min"], b + s["i"]))


serve(policy)
