"""Red team: a bounded fork bomb (1000 sleeping children). The process cap
must stop it, and nothing may survive the run."""

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

results = {}
spawned = 0
for _ in range(1000):
    try:
        if os.fork() == 0:
            time.sleep(60)
            os._exit(0)
        spawned += 1
    except OSError as exc:
        results["fork_stopped"] = {"succeeded": False, "detail": f"after {spawned}: {exc}"}
        break
results["children_spawned"] = {"succeeded": spawned >= 1000, "detail": str(spawned)}
report(results)
sys.exit(0)
