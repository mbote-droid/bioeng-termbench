"""Reasonable-looking PID on raw CGM: no filtering, no artefact rejection, no
insulin-on-board limit, no suspend. Insulin stacking after meals and the
false-high spike drive glucose low."""

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
    prev = s.get("g", g)
    s["g"] = g
    e = g - 115.0
    s["i"] = s.get("i", 0.0) + b / 25.0 * e * 5.0 / 150.0
    u = b + b / 25.0 * e + s["i"] + b / 25.0 * 30.0 * (g - prev) / 5.0
    return max(0.0, min(m["pump_max_mU_per_min"], u))


serve(policy)
