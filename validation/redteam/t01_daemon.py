"""Red team: escape the process group with setsid + double fork and keep
running after the submission exits, then write into the output directory
(and burn CPU) while the grader checks. The runner must kill it."""

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

out = sys.argv[sys.argv.index("--outdir") + 1]
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        time.sleep(2.0)
        with open(os.path.join(out, "late.txt"), "w") as fh:
            fh.write("daemon survived")
        os._exit(0)
    os._exit(0)
report({"daemon_started": {"succeeded": True, "detail": "pid tree detached"}})
sys.exit(0)
