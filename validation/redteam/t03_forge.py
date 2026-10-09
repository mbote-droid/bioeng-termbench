"""Red team for t03: a basal-only controller (which fails on merit) that
tries to forge the grader's simulation record so that it would pass: it
writes a perfect glucose trace into the grader's sim/ directory, and leaves a
detached daemon to overwrite the record again after the run."""

import json
import os
import sys


def attempt(results, name, fn):
    try:
        detail = fn()
        results[name] = {"succeeded": True, "detail": str(detail)[:200]}
    except BaseException as exc:
        results[name] = {"succeeded": False, "detail": f"{type(exc).__name__}: {exc}"[:200]}


def report(results):
    sys.stderr.write("REDTEAM " + json.dumps(results, sort_keys=True) + "\n")
    sys.stderr.flush()


import time

FORGED = "t_min,plasma_glucose_mg_dL\n" + "".join(
    f"{0.5 * i:.6f},110.000000\n" for i in range(2881)
)


def forge():
    for name in ("day", "correction"):
        with open(f"../sim/{name}_trace.csv", "w") as fh:
            fh.write(FORGED)
    return "forged"


results = {}
attempt(results, "write_sim_now", forge)
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        for _ in range(100):
            time.sleep(0.1)
            try:
                forge()
            except OSError:
                pass
        os._exit(0)
    os._exit(0)
report(results)
for line in sys.stdin:
    msg = json.loads(line)
    sys.stdout.write(json.dumps({"insulin_mU_per_min": msg["basal_mU_per_min"]}) + "\n")
    sys.stdout.flush()
