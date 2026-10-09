"""Crashes on the first missing CGM reading (None arithmetic)."""

import json
import sys


def serve(policy):
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        sys.stdout.write(json.dumps({"insulin_mU_per_min": policy(msg, state)}) + "\n")
        sys.stdout.flush()


serve(lambda m, s: m["basal_mU_per_min"] * (1 + (m["cgm_mg_dL"] - 115) / 100))
