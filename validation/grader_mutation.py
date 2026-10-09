"""Mutation testing of the graders (hypothesis H2: the test suite would
notice if a grader were wrong).

Two mutation operators on a task's grader module:

  check_off   one check can never fail (its Outcome is forced to pass).
              Survival means no test depends on that check: either the check
              is redundant or a negative control is missing.
  threshold   one module-level numeric constant is halved or doubled.
              Survival means no test sits close enough to that boundary to
              notice; expected for timeouts and sizes, informative for
              acceptance thresholds.

Each mutant is written to a copy of the package that shadows the installed
one (PYTHONPATH), and the task's tests run against it; a failing test kills
the mutant. Results go to results/validation/mutation/<task>.json.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    # Boundary tests come first so a killed mutant stops early (-x).
    "t01-ecg-denoise-rpeaks": (
        "t01_ecg.py",
        ["tests/tasks/test_t01_thresholds.py", "tests/tasks/test_t01_negative_controls.py",
         "tests/tasks/test_t01_references.py"],
        ("check_off", "threshold"),
    ),
    "t02-pk-fit-dose": (
        "t02_pk.py",
        ["tests/tasks/test_t02_thresholds.py", "tests/tasks/test_t02.py", "-m", "not slow"],
        ("check_off", "threshold"),
    ),
    "t03-glucose-closed-loop": (
        "t03_glucose.py",
        ["tests/tasks/test_t03_thresholds.py", "tests/tasks/test_t03.py", "-m", "not octave"],
        ("check_off", "threshold"),
    ),
    "t04-bioheat-ablation": (
        "t04_ablation.py",
        ["tests/tasks/test_t04_thresholds.py", "tests/tasks/test_t04.py", "-k",
         "threshold or boundary or accuracy or mutant or inputs or order or robust or contract"],
        ("check_off", "threshold"),
    ),
    "t05-pk-debug": (
        "t05_pk_debug.py",
        ["tests/tasks/test_t05_thresholds.py", "tests/tasks/test_t05.py", "-m", "not slow"],
        ("check_off", "threshold"),
    ),
}


@dataclass(frozen=True)
class Mutant:
    operator: str
    target: str
    description: str
    source: str
    role: str = ""  # threshold mutants: "acceptance" (used by a check) or "configuration"


def _names_used_by_checks(tree: ast.Module) -> set[str]:
    """Constants read inside check functions or check factories: acceptance thresholds."""
    used: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (
            node in _check_functions(tree) or node.name in {"scenario_check", "_convergence"}
            or node.name.startswith("_time_fraction")
        ):
            used |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    return used


def _check_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "check":
                    out.append(node)
    return out


def _force_pass(fn: ast.FunctionDef) -> ast.FunctionDef:
    """The check can never fail: the original body still runs (so side effects
    such as loading outputs for later checks are kept), but any outcome or
    exception is replaced by a pass. Rewriting only `Outcome(...)` calls would
    miss checks that delegate to a helper or fail by raising, which made such
    mutants equivalent to the original by construction."""
    fn = ast.parse(ast.unparse(fn)).body[0]  # deep copy
    assert isinstance(fn, ast.FunctionDef)
    inner = ast.FunctionDef(
        name="_original", args=fn.args, body=fn.body, decorator_list=[], returns=None,
        type_params=[],
    )
    wrapper = ast.parse(
        "try:\n    _original(ctx)\nexcept Exception:\n    pass\nreturn Outcome(True)"
    ).body
    fn.body = [inner, *wrapper]
    return ast.fix_missing_locations(fn)


def mutants(path: Path, operators: tuple[str, ...]) -> list[Mutant]:
    src = path.read_text()
    tree = ast.parse(src)
    out: list[Mutant] = []
    if "check_off" in operators:
        for fn in _check_functions(tree):
            code = next(d.args[0].value for d in fn.decorator_list if isinstance(d, ast.Call))
            new_tree = ast.parse(src)
            for i, node in enumerate(new_tree.body):
                if isinstance(node, ast.FunctionDef) and node.name == fn.name:
                    new_tree.body[i] = _force_pass(node)
            out.append(Mutant("check_off", code, f"{code} never fails", ast.unparse(new_tree)))
    factory = "    def fn(ctx: CheckContext) -> Outcome:\n"
    if "check_off" in operators and factory in src and "SCENARIO_CODES" in src:
        # Checks built by a factory (t05): switch off one scenario at a time.
        codes = ast.literal_eval(
            next(
                n.value
                for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SCENARIO_CODES"
            )
        )
        for name, code in codes:
            new_src = src.replace(
                factory,
                factory + f"        if name == {name!r}:\n            return Outcome(True)\n",
                1,
            )
            out.append(Mutant("check_off", code, f"{code} never fails", new_src))
    if "threshold" in operators:
        acceptance = _names_used_by_checks(tree)
        lines = src.splitlines(keepends=True)
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            name = getattr(node.targets[0], "id", "")
            if not name.isupper() or not isinstance(node.value, ast.Constant):
                continue
            value = node.value.value
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            for factor in (0.5, 2.0):
                new_value = (
                    value * factor if isinstance(value, float) else max(1, round(value * factor))
                )
                line = lines[node.lineno - 1]
                seg = ast.get_source_segment(src, node.value)
                assert seg is not None
                mutated = line.replace(seg, repr(new_value), 1)
                new_src = "".join([*lines[: node.lineno - 1], mutated, *lines[node.lineno :]])
                role = "acceptance" if name in acceptance else "configuration"
                out.append(Mutant("threshold", name, f"{name} {value!r} -> {new_value!r}", new_src, role))
    return out


def run_tests(pkg_root: Path, test_args: list[str], timeout_s: float) -> tuple[bool, float]:
    env = dict(os.environ, PYTHONPATH=str(pkg_root), PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *test_args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        killed = proc.returncode != 0
    except subprocess.TimeoutExpired:
        killed = True  # a hang is a detected change
    return killed, time.time() - t0


def main(task: str, out: str | None, only: tuple[str, ...] | None = None) -> int:
    """Run the mutants; with `only`, rerun those operators and merge into the
    existing result file."""
    module, test_args, operators = CONFIG[task]
    if only:
        operators = tuple(o for o in operators if o in only)
    module_path = ROOT / "src/bioengbench/tasks" / module
    work = Path(tempfile.mkdtemp(prefix="bioeng-mut-"))
    pkg_root = work / "src"
    shutil.copytree(ROOT / "src", pkg_root, ignore=shutil.ignore_patterns("__pycache__"))
    target = pkg_root / "bioengbench/tasks" / module
    baseline_killed, base_s = run_tests(pkg_root, test_args, 3600)
    if baseline_killed:
        print(f"{task}: tests fail on the unmutated grader; aborting", file=sys.stderr)
        return 1
    results = []
    try:
        for m in mutants(module_path, operators):
            target.write_text(m.source)
            killed, secs = run_tests(pkg_root, test_args, max(120.0, 5 * base_s))
            results.append(
                {
                    "operator": m.operator,
                    "role": m.role,
                    "target": m.target,
                    "description": m.description,
                    "killed": killed,
                    "seconds": round(secs, 1),
                }
            )
            print(f"{task} {m.description:45s} {'killed' if killed else 'SURVIVED'}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    dest = Path(out) if out else ROOT / "results/validation/mutation" / f"{task}.json"
    if only and dest.exists():  # merge: keep the other operators' results
        previous = json.loads(dest.read_text())["mutants"]
        results = [r for r in previous if r["operator"] not in operators] + results
    by_op = {}
    present = {r["operator"] for r in results}
    for op, role in (("check_off", ""), ("threshold", "acceptance"), ("threshold", "configuration")):
        if op not in present:
            continue
        rs = [r for r in results if r["operator"] == op and r["role"] == role]
        key = op if not role else f"threshold_{role}"
        by_op[key] = {"total": len(rs), "killed": sum(r["killed"] for r in rs)}
    report = {
        "task": task,
        "module": module,
        "tests": test_args,
        "summary": by_op,
        "mutants": results,
        "provenance": provenance.collect(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(CONFIG))
    ap.add_argument("out", nargs="?")
    ap.add_argument("--only", nargs="+", choices=("check_off", "threshold"),
                    help="rerun these operators and merge into the existing result file")
    a = ap.parse_args()
    raise SystemExit(main(a.task, a.out, tuple(a.only) if a.only else None))
