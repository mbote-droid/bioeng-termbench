"""Run the red-team submissions (validation/redteam/) against the real
grader and check that every attack is contained.

Must run as root with a `sandbox` user, as in the verifier image. Each attack
is graded twice: sandboxed (the supported configuration; every attack must
be contained) and unsandboxed (to record which protections depend on the
sandbox user; expected to show some attacks succeeding). A stand-in /tests
directory (mode 0700) and a root-only canary file reproduce what the verifier
image keeps away from submissions.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance
from campaign import campaign_seeds, python_for_submissions

from bioengbench.core import registry
from bioengbench.core.runner import LocalRunner
from bioengbench.core.scoring import Status

HERE = Path(__file__).resolve().parent / "redteam"
SANDBOX = "sandbox"
CANARY = Path("/root/redteam_canary.txt")
TESTS = Path("/tests")


def _uid_processes(uid: int) -> list[int]:
    pids = []
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            for line in status.read_text().splitlines():
                if line.startswith("Uid:"):
                    if int(line.split()[1]) == uid:
                        pids.append(int(status.parent.name))
                    break
        except OSError:
            continue
    return pids


def _kill_matching(fragment: str) -> int:
    """Clean up after unsandboxed runs: kill processes whose command line
    mentions the run's temporary directory."""
    killed = 0
    for cmd in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if fragment in cmd.read_bytes().decode(errors="ignore"):
                os.kill(int(cmd.parent.name), signal.SIGKILL)
                killed += 1
        except (OSError, ProcessLookupError):
            continue
    return killed


def _redteam_lines(case_dir: Path) -> dict:
    out: dict = {}
    for log in case_dir.rglob("*.log"):
        for line in log.read_text(errors="ignore").splitlines():
            if line.startswith("REDTEAM "):
                out.update(json.loads(line[len("REDTEAM ") :]))
    return out


def grade_attack(task_id: str, attack: str, sandboxed: bool) -> dict:
    task = registry.get(task_id)
    work = Path(tempfile.mkdtemp(prefix=f"redteam-{attack}-"))
    work.chmod(0o755)
    sub = work / "sub" / f"{attack}.py"
    sub.parent.mkdir()
    shutil.copyfile(HERE / f"{attack}.py", sub)
    runner = LocalRunner((python_for_submissions(),), sandbox_user=SANDBOX if sandboxed else None)
    seed = campaign_seeds(f"redteam/{attack}", 1)[0]
    t0 = time.time()
    case = task.grade_case(sub, seed, runner, work / "cases")
    elapsed = time.time() - t0
    time.sleep(3.0)  # give any surviving daemon time to act
    case_dir = next((work / "cases").glob("case-*"))
    uid = pwd.getpwnam(SANDBOX).pw_uid
    record = {
        "attack": attack,
        "task": task_id,
        "sandboxed": sandboxed,
        "seconds": round(elapsed, 1),
        "passed": case.passed,
        "failed_codes": sorted(r.code for r in case.results if r.status is Status.FAIL),
        "run_message": next((r.message for r in case.results if r.code == "RUN"), ""),
        "attempts": _redteam_lines(case_dir),
        "survivors": len(_uid_processes(uid)) if sandboxed else _kill_matching(str(work)),
        "late_file": any(case_dir.rglob("late.txt")),
        "case_dir_files": sorted(p.name for p in case_dir.iterdir()),
    }
    if task_id.startswith("t03"):
        trace = (case_dir / "sim" / "day_trace.csv").read_text().splitlines()[1:50]
        record["sim_forged"] = all(line.endswith(",110.000000") for line in trace)
    big = list(case_dir.rglob("big.bin"))
    record["largest_output_mb"] = round(max((p.stat().st_size for p in big), default=0) / 2**20, 1)
    shutil.rmtree(work, ignore_errors=True)
    return record


def contained(r: dict) -> tuple[bool, str]:
    """Is the attack contained? (Only meaningful for sandboxed runs.)"""
    a = r["attack"]
    if r["passed"]:
        return False, "attack earned a passing verdict"
    if r["survivors"]:
        return False, f"{r['survivors']} sandbox processes survived the run"
    if r["late_file"]:
        return False, "a detached process wrote after the run"
    leaked = [k for k, v in r["attempts"].items() if v["succeeded"] and k not in {"daemon_started"}]
    if a == "t01_fork_bomb":
        n = int(r["attempts"].get("children_spawned", {}).get("detail", "0"))
        return n < 1000, f"fork bomb spawned {n} children"
    if leaked:
        return False, f"succeeded: {leaked}"
    if a == "t01_symlink" and "symbolic link" not in r["run_message"]:
        return False, "symlinked outputs were not rejected"
    if a == "t03_forge" and r.get("sim_forged"):
        return False, "simulation record forged"
    if a == "t01_disk_fill" and r["largest_output_mb"] > 256:
        return False, "file-size limit not applied"
    return True, "contained"


ATTACKS = [
    ("t01-ecg-denoise-rpeaks", "t01_escape"),
    ("t01-ecg-denoise-rpeaks", "t01_symlink"),
    ("t01-ecg-denoise-rpeaks", "t01_fork_bomb"),
    ("t01-ecg-denoise-rpeaks", "t01_daemon"),
    ("t01-ecg-denoise-rpeaks", "t01_disk_fill"),
    ("t01-ecg-denoise-rpeaks", "t01_memory"),
    ("t03-glucose-closed-loop", "t03_forge"),
]


def main(out: str) -> int:
    if os.geteuid() != 0:
        print("red team needs root (as in the verifier) and a 'sandbox' user", file=sys.stderr)
        return 2
    pwd.getpwnam(SANDBOX)
    made_tests = not TESTS.exists()
    if made_tests:
        TESTS.mkdir(mode=0o700)
        (TESTS / "dev_seeds.txt").write_text("deadbeef\n")
    CANARY.write_text("secret\n")
    CANARY.chmod(0o600)
    results = []
    try:
        for task_id, attack in ATTACKS:
            for sandboxed in (True, False):
                r = grade_attack(task_id, attack, sandboxed)
                ok, why = contained(r)
                r["contained"], r["verdict"] = ok, why
                results.append(r)
                tag = "sandboxed  " if sandboxed else "unsandboxed"
                print(f"{attack:15s} {tag} contained={ok!s:5s} {why}")
    finally:
        CANARY.unlink(missing_ok=True)
        if made_tests:
            shutil.rmtree(TESTS)
    report = {
        "results": results,
        "all_sandboxed_contained": all(r["contained"] for r in results if r["sandboxed"]),
        "provenance": provenance.collect({"submission_python": python_for_submissions()}),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["all_sandboxed_contained"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results/validation/redteam.json"))
