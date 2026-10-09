"""Red team: allocate 8 GiB. The address-space limit must stop it."""

import json
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


results = {}
attempt(results, "allocate_8GiB", lambda: len(bytearray(8 << 30)))
report(results)
sys.exit(1)
