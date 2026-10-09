"""Exceeds the per-step deadline after a while."""

import json
import sys
import time


def serve(policy):
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        sys.stdout.write(json.dumps({"insulin_mU_per_min": policy(msg, state)}) + "\n")
        sys.stdout.flush()


def policy(m, s):
    if m["t_min"] >= 60:
        time.sleep(5)
    return m["basal_mU_per_min"]


serve(policy)
