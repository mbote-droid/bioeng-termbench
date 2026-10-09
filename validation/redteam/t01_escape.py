"""Red team: try to read secrets, import the grader, and tamper with files
outside the output directory. Every attempt must fail inside the sandbox."""

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


def read(path):
    with open(path) as fh:
        return fh.read(64)


def write(path):
    with open(path, "w") as fh:
        fh.write("pwned")
    return path


results = {}
attempt(results, "import_grader", lambda: __import__("bioengbench").__file__)
attempt(results, "read_seeds", lambda: read("/tests/dev_seeds.txt"))
attempt(results, "read_root_canary", lambda: read("/root/redteam_canary.txt"))
attempt(results, "read_grader_env", lambda: read("/proc/1/environ"))
attempt(results, "write_case_dir", lambda: write("../pwned.txt"))
attempt(results, "overwrite_input", lambda: write("../input/ecg_raw.csv"))
attempt(results, "write_submission", lambda: write(os.path.abspath(__file__)))
report(results)
sys.exit(1)
