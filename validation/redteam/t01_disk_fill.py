"""Red team: fill the disk (1 GiB file). The file-size limit must stop it."""

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


out = sys.argv[sys.argv.index("--outdir") + 1]
os.makedirs(out, exist_ok=True)
results = {}


def fill():
    block = b"x" * (1 << 20)
    with open(os.path.join(out, "big.bin"), "wb") as fh:
        for _ in range(1024):
            fh.write(block)
    return "1 GiB written"


import signal

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # turn the kill into an OSError we can report
attempt(results, "write_1GiB", fill)
report(results)
sys.exit(1)
