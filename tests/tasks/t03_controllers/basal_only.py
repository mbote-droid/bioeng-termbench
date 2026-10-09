"""Open loop: basal rate only. Must fail on performance (meals uncovered)."""

import json
import sys


def serve(policy):
    state = {}
    for line in sys.stdin:
        msg = json.loads(line)
        sys.stdout.write(json.dumps({"insulin_mU_per_min": policy(msg, state)}) + "\n")
        sys.stdout.flush()


serve(lambda m, s: m["basal_mU_per_min"])
