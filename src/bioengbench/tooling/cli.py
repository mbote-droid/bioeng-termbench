"""`bioeng` command-line interface: one entry point for every terminal workflow.

    bioeng list
    bioeng seeds new N [--out FILE]
    bioeng make-inputs TASK --seed HEX --outdir DIR
    bioeng check-outputs TASK --seed HEX --outdir DIR [--verdict FILE]
    bioeng grade TASK --submission PATH --seeds FILE [--lang python|octave|julia]
                 [--runner local|docker] [--sandbox-user USER] [--trusted] [--python EXE]
                 [--image IMAGE]
                 [--reward-dir DIR] [--verdict FILE] [--workdir DIR]

Exit codes: 0 graded and passed, 1 graded and failed, 2 usage or setup error.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from bioengbench.core import registry
from bioengbench.core.runner import DockerRunner, LocalRunner, Runner
from bioengbench.core.scoring import CaseResult, Verdict
from bioengbench.core.seeds import from_hex, new_seed, read_seed_file, to_hex

INTERPRETERS: dict[str, tuple[str, ...]] = {
    "octave": ("octave-cli", "--no-gui", "--quiet"),
    "julia": ("julia", "--startup-file=no", "--history-file=no"),
}


def interpreter(lang: str, python: str | None = None) -> tuple[str, ...]:
    if lang == "python":
        return (python or sys.executable,)
    return INTERPRETERS[lang]


def _cmd_list(_: argparse.Namespace) -> int:
    for task_id in registry.task_ids():
        task = registry.get(task_id)
        print(f"{task_id}\tv{task.version}")
    return 0


def _cmd_seeds(args: argparse.Namespace) -> int:
    lines = [to_hex(new_seed()) for _ in range(args.n)]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def _cmd_make_inputs(args: argparse.Namespace) -> int:
    task = registry.get(args.task)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    task.write_inputs(task.make_case(from_hex(args.seed)), outdir)
    return 0


def _cmd_check_outputs(args: argparse.Namespace) -> int:
    """Grade outputs produced elsewhere (another language, another machine) for one seed.

    Single-run tasks only: closed-loop and multi-run tasks need the grader to
    drive the submission, so they go through `grade`.
    """
    task = registry.get(args.task)
    seed = from_hex(args.seed)
    case = task.make_case(seed)
    if len(task.run_plans(case)) != 1:
        raise ValueError(f"{task.task_id} needs `bioeng grade`; it runs the submission itself")
    results = task.evaluate_outputs(Path(args.outdir), case)
    verdict = Verdict(task.task_id, task.version, (CaseResult(seed, results),))
    if args.verdict:
        Path(args.verdict).write_text(verdict.to_json())
    for r in results:
        print(f"{r.code:20s} {r.status}  {r.message}")
    print(f"{task.task_id} v{task.version}: {'PASS' if verdict.passed else 'FAIL'}")
    return 0 if verdict.passed else 1


def _runner(args: argparse.Namespace) -> Runner:
    if args.runner == "docker":
        if not args.image:
            raise SystemExit("--runner docker needs --image")
        prefix = ("python",) if args.lang == "python" else INTERPRETERS[args.lang]
        return DockerRunner(args.image, prefix)
    return LocalRunner(interpreter(args.lang, args.python), sandbox_user=args.sandbox_user)


UNSANDBOXED_WARNING = (
    "warning: grading without --sandbox-user. The submission runs as you and can "
    "read and write everything you can, including the grader's records (the red "
    "team forges a passing t03 verdict this way). Use only for trusted code; use "
    "--sandbox-user (as root) or --runner docker for anything else."
)


def _check_isolation(args: argparse.Namespace) -> str | None:
    """Refuse the one configuration that is never safe: root without a sandbox user."""
    if args.runner != "local" or args.sandbox_user:
        return None
    if os.geteuid() == 0 and not args.trusted:
        return (
            "refusing to run a submission as root without --sandbox-user "
            "(pass --trusted to override)"
        )
    print(UNSANDBOXED_WARNING, file=sys.stderr)
    return None


def _cmd_grade(args: argparse.Namespace) -> int:
    if problem := _check_isolation(args):
        print(problem, file=sys.stderr)
        return 2
    task = registry.get(args.task)
    seeds = read_seed_file(Path(args.seeds))
    submission = Path(args.submission).resolve()
    if not submission.exists():
        print(f"submission not found: {submission}", file=sys.stderr)
        verdict_dir = Path(args.reward_dir) if args.reward_dir else None
        if verdict_dir:
            verdict_dir.mkdir(parents=True, exist_ok=True)
            (verdict_dir / "reward.txt").write_text("0\n")
        return 1
    with tempfile.TemporaryDirectory(prefix="bioeng-grade-", dir=args.workdir) as tmp:
        Path(tmp).chmod(0o755)
        verdict = task.grade(submission, seeds, _runner(args), Path(tmp), args.lang)
    if args.reward_dir:
        verdict.write_reward(Path(args.reward_dir))
    if args.verdict:
        Path(args.verdict).write_text(verdict.to_json())
    for case in verdict.cases:
        status = "PASS" if case.passed else "FAIL"
        detail = ", ".join(sorted({r.code for r in case.failures()}))
        print(f"case {to_hex(case.seed)[:12]}  {status}  {detail}")
    print(f"{task.task_id} v{task.version}: {'PASS' if verdict.passed else 'FAIL'}")
    return 0 if verdict.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bioeng", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered tasks").set_defaults(fn=_cmd_list)

    s = sub.add_parser("seeds", help="seed utilities")
    s_sub = s.add_subparsers(dest="seeds_command", required=True)
    s_new = s_sub.add_parser("new", help="print N fresh 128-bit seeds")
    s_new.add_argument("n", type=int)
    s_new.add_argument("--out")
    s_new.set_defaults(fn=_cmd_seeds)

    m = sub.add_parser("make-inputs", help="write agent-visible inputs for one seed")
    m.add_argument("task")
    m.add_argument("--seed", required=True)
    m.add_argument("--outdir", required=True)
    m.set_defaults(fn=_cmd_make_inputs)

    c = sub.add_parser("check-outputs", help="grade precomputed outputs for one seed")
    c.add_argument("task")
    c.add_argument("--seed", required=True)
    c.add_argument("--outdir", required=True)
    c.add_argument("--verdict")
    c.set_defaults(fn=_cmd_check_outputs)

    g = sub.add_parser("grade", help="grade a submission on hidden seeds")
    g.add_argument("task")
    g.add_argument("--submission", required=True)
    g.add_argument("--seeds", required=True)
    g.add_argument("--lang", choices=["python", "octave", "julia"], default="python")
    g.add_argument("--runner", choices=["local", "docker"], default="local")
    g.add_argument("--sandbox-user")
    g.add_argument(
        "--trusted",
        action="store_true",
        help="allow running as root without a sandbox user (your own reference code only)",
    )
    g.add_argument("--python", help="interpreter for Python submissions (default: this one)")
    g.add_argument("--image")
    g.add_argument("--reward-dir")
    g.add_argument("--verdict")
    g.add_argument("--workdir")
    g.set_defaults(fn=_cmd_grade)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.fn(args)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
