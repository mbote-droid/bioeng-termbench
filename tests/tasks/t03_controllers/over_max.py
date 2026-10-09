"""Requests twice the pump maximum when high: the pump saturates, but asking
for an impossible rate is an interface violation (ACTUATOR)."""

import json
import sys


def serve(policy):
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        sys.stdout.write(json.dumps({"insulin_mU_per_min": policy(msg, state)}) + "\n")
        sys.stdout.flush()


def policy(m, s):
    g = m["cgm_mg_dL"]
    return 2 * m["pump_max_mU_per_min"] if g is not None and g > 240 else m["basal_mU_per_min"]


serve(policy)
