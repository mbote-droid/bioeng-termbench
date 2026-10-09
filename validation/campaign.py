"""Grader validation campaign.

    python validation/campaign.py probe t03-glucose-closed-loop/naive_pid --seeds 40
    python validation/campaign.py redteam
    python validation/campaign.py report

`probe` grades one probe (see probes.py) on campaign seeds and writes
results/validation/<task>/<probe>.json. Seeds are derived from a fixed
campaign master seed and the probe key, so a rerun grades the same cases and
must produce identical verdicts. The Snakefile runs every probe in parallel
and then `report`.

Statistics. For a reference, every failing case is a false reject: we report
failures/n with a Wilson 95 % interval, and the rule-of-three style bound when
there are none. For a negative probe, every passing case is a false accept
per case; the task reward needs every hidden case to pass, so with k hidden
cases the chance a probe earns reward is p^k (reported for k = 8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probes as P
import provenance

from bioengbench.checks.stats import wilson_interval, zero_failure_upper_bound
from bioengbench.core import registry
from bioengbench.core.runner import LocalRunner
from bioengbench.core.scoring import Status
from bioengbench.core.seeds import to_hex
from bioengbench.tooling.cli import interpreter

MASTER_SEED = 0xB10E_46A1_7E2C_0D5B
HIDDEN_CASES = 8
RESULTS = P.ROOT / "results" / "validation"
DEFAULT_SEEDS = {
    P.T01: 100,
    P.T02: 30,
    P.T03: 300,
    P.T04: 30,
    P.T05: 100,
}
AGENT_PYTHON = Path("/opt/agent-python/bin/python")


def campaign_seeds(key: str, n: int) -> list[int]:
    """128-bit seeds, a pure function of the master seed and the probe key."""
    digest = hashlib.sha256(key.encode()).digest()
    ss = np.random.SeedSequence(MASTER_SEED, spawn_key=tuple(digest[:8]))
    return [int.from_bytes(c.generate_state(4, np.uint32).tobytes(), "little") for c in ss.spawn(n)]


def python_for_submissions() -> str:
    """Prefer an interpreter with exactly the agent image's packages (no
    bioengbench), as in the Docker verifier; fall back to this one."""
    return str(AGENT_PYTHON) if AGENT_PYTHON.exists() else sys.executable


def run_probe(key: str, n: int | None, workroot: Path | None = None) -> dict:
    probe = P.by_key()[key]
    task = registry.get(probe.task)
    n = n or probe.seeds or DEFAULT_SEEDS[probe.task]
    seeds = campaign_seeds(key, n)
    work = Path(tempfile.mkdtemp(prefix="bioeng-val-", dir=workroot))
    runner = LocalRunner(interpreter(probe.lang, python_for_submissions()))
    t0 = time.time()
    try:
        sub = probe.build(work)
        cases = [task.grade_case(sub, s, runner, work / "cases", probe.lang) for s in seeds]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    elapsed = time.time() - t0
    passed = sum(c.passed for c in cases)
    fail_codes = Counter(r.code for c in cases for r in c.results if r.status is Status.FAIL)
    margins: dict[str, dict[str, float | None]] = {}
    for c in cases:
        for r in c.results:
            if r.measured is None:
                continue
            m = margins.setdefault(
                r.code, {"min": r.measured, "max": r.measured, "threshold": r.threshold}
            )
            m["min"] = min(m["min"], r.measured)
            m["max"] = max(m["max"], r.measured)
    result = {
        "probe": key,
        "task": probe.task,
        "task_version": task.version,
        "kind": probe.kind,
        "note": probe.note,
        "lang": probe.lang,
        "expected_codes": sorted(probe.expected),
        "n_cases": n,
        "n_passed": passed,
        "fail_codes": dict(sorted(fail_codes.items())),
        "margins": margins,
        "seconds": round(elapsed, 1),
        "seeds": [to_hex(s) for s in seeds],
        "case_verdicts_sha256": hashlib.sha256(
            json.dumps([c.to_dict() for c in cases], sort_keys=True).encode()
        ).hexdigest(),
    }
    result["assessment"] = assess(result)
    result["provenance"] = provenance.collect({"submission_python": python_for_submissions()})
    return result


def assess(r: dict) -> dict:
    n, k = r["n_cases"], r["n_passed"]
    if r["kind"] == "reference":
        fails = n - k
        lo, hi = wilson_interval(fails, n)
        return {
            "ok": fails == 0,
            "false_reject_rate": fails / n,
            "false_reject_95ci": [lo, hi],
            "false_reject_upper_95": zero_failure_upper_bound(n) if fails == 0 else hi,
        }
    p = k / n
    lo, hi = wilson_interval(k, n)
    expected_fired = set(r["expected_codes"]) <= set(r["fail_codes"])
    if r["kind"] == "boundary":
        # A method near a contract limit may pass some cases and fail others;
        # what must hold is that it never fails on any other clause.
        only_expected = set(r["fail_codes"]) <= set(r["expected_codes"])
        return {"ok": only_expected, "fails_only_on_expected_codes": only_expected,
                "per_case_accept": p, "per_case_accept_95ci": [lo, hi]}
    return {
        "ok": expected_fired and hi**HIDDEN_CASES < 0.05,
        "expected_codes_fired": expected_fired,
        "per_case_accept": p,
        "per_case_accept_95ci": [lo, hi],
        f"reward_probability_{HIDDEN_CASES}_cases_upper": hi**HIDDEN_CASES,
    }


def cmd_probe(a: argparse.Namespace) -> int:
    result = run_probe(a.key, a.seeds)
    out = Path(a.out) if a.out else RESULTS / f"{a.key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    s = result["assessment"]
    print(
        f"{a.key}: {result['n_passed']}/{result['n_cases']} passed, ok={s['ok']}, {result['seconds']} s"
    )
    return 0


def cmd_reassess(_: argparse.Namespace) -> int:
    """Re-apply the current expectations (kind, expected codes, note) and
    assessment rules to existing result files. Grading outcomes are not
    touched; probes whose submission or seeds changed must be rerun instead."""
    registry_ = P.by_key()
    for path in sorted(RESULTS.glob("t0*/*.json")):
        r = json.loads(path.read_text())
        probe = registry_.get(r["probe"])
        if probe is None:
            continue
        r.update(kind=probe.kind, note=probe.note, expected_codes=sorted(probe.expected))
        r["assessment"] = assess(r)
        path.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
        print(f"{r['probe']}: ok={r['assessment']['ok']}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    for p in P.PROBES:
        print(p.key)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("probe")
    s.add_argument("key")
    s.add_argument("--seeds", type=int)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_probe)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("reassess").set_defaults(fn=cmd_reassess)
    s = sub.add_parser("redteam")
    s.add_argument("--out", default=str(RESULTS / "redteam.json"))
    s.set_defaults(fn=lambda a: __import__("redteam_runner").main(a.out))
    s = sub.add_parser("mutation")
    s.add_argument("task")
    s.add_argument("--out")
    s.set_defaults(fn=lambda a: __import__("grader_mutation").main(a.task, a.out))
    s = sub.add_parser("report")
    s.add_argument("--out", default=str(P.ROOT / "docs" / "VALIDATION_REPORT.md"))
    s.set_defaults(fn=lambda a: __import__("report").main(a.out))
    a = ap.parse_args()
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
