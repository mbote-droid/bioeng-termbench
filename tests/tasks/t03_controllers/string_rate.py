"""Replies with the rate as a string."""

import json
import sys


def serve(policy):
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        sys.stdout.write(json.dumps({"insulin_mU_per_min": policy(msg, state)}) + "\n")
        sys.stdout.flush()


serve(lambda m, s: str(m["basal_mU_per_min"]))
