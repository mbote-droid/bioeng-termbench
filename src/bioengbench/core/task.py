"""Task plugin contract (Template Method).

`grade` fixes the lifecycle for every task:

    for each hidden seed:
        make_case(seed)            task: simulate data and ground truth
        write_inputs(case, dir)    task: write only what the agent may see
        execute(...)               core (default) or task (closed loop): run the
                                   submission in a sandbox, once per RunPlan
        checks().evaluate(ctx)     task: compare outputs with truth

Tasks fill in the steps; they never touch sandboxing or scoring, and the core
never knows a task's physics.

Run plans. Most tasks run the submission once. Some need more: t04 reruns the
solver at refined resolutions to measure its order of accuracy. Each RunPlan
writes to its own output directory, exposed to checks as ctx.run_dirs[name].

Languages. A task declares the command line for each language it accepts
(`argv(lang)`); the interpreter itself is chosen by the caller's Runner.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from bioengbench.checks.base import CheckContext, CheckGroup
from bioengbench.core.runner import Runner, RunSpec
from bioengbench.core.scoring import CaseResult, CheckResult, Status, Verdict
from bioengbench.core.seeds import to_hex

RUN_CODE = "RUN"
MAIN = "main"


@dataclass(frozen=True)
class RunPlan:
    name: str
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Execution:
    """Outcome of running the submission for one case."""

    ok: bool
    reason: str = ""


def symlinks_in(directory: Path) -> list[str]:
    """Symbolic links under `directory` (not followed). Outputs must be real
    files: a link could point the grader at a file the submission cannot read
    itself, or at a file outside the case directory."""
    if not directory.exists():
        return []
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_symlink())


def stage_submission(submission: Path, dest_dir: Path) -> Path:
    """Copy a file or package directory to a world-readable location.

    The unprivileged sandbox user then never needs access to the original path,
    which may sit under a root-only directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.chmod(0o755)
    staged = dest_dir / submission.name
    if submission.is_dir():
        shutil.copytree(submission, staged, ignore=shutil.ignore_patterns("__pycache__"))
        for path in [staged, *staged.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
    else:
        shutil.copyfile(submission, staged)
        staged.chmod(0o644)
    return staged


class TaskPlugin(ABC):
    task_id: ClassVar[str]
    version: ClassVar[str]  # semantic version; bump on any grader change
    timeout_s: ClassVar[float] = 60.0
    memory_mb: ClassVar[int] = 2048
    extra_env: ClassVar[tuple[tuple[str, str], ...]] = ()

    @abstractmethod
    def make_case(self, seed: int) -> Any:
        """Simulate one case and its ground truth, fully determined by `seed`."""

    @abstractmethod
    def write_inputs(self, case: Any, input_dir: Path) -> None:
        """Write the agent-visible inputs for `case`. Never write truth here."""

    @abstractmethod
    def argv(self, lang: str = "python") -> tuple[str, ...]:
        """Command line for `lang`, using {submission}, {submission_dir},
        {input_dir} and {output_dir} placeholders."""

    @abstractmethod
    def checks(self) -> CheckGroup:
        """The grading suite for one case."""

    def languages(self) -> tuple[str, ...]:
        return ("python",)

    def run_plans(self, case: Any) -> tuple[RunPlan, ...]:
        return (RunPlan(MAIN),)

    # -- execution (overridable for closed-loop tasks) ---------------------------

    def spec(
        self, staged: Path, input_dir: Path, output_dir: Path, lang: str, plan: RunPlan
    ) -> RunSpec:
        return RunSpec(
            argv=(*self.argv(lang), *plan.extra_args),
            submission=staged,
            input_dir=input_dir,
            output_dir=output_dir,
            timeout_s=self.timeout_s,
            memory_mb=self.memory_mb,
            extra_env=self.extra_env,
        )

    def execute(
        self, case: Any, staged: Path, runner: Runner, case_dir: Path, lang: str
    ) -> tuple[Execution, dict[str, Path]]:
        """Run every RunPlan; stop at the first failure."""
        input_dir = case_dir / "input"
        run_dirs: dict[str, Path] = {}
        for plan in self.run_plans(case):
            out = case_dir / ("output" if plan.name == MAIN else f"output-{plan.name}")
            run_dirs[plan.name] = out
            result = runner.run(self.spec(staged, input_dir, out, lang, plan))
            if not result.ok:
                reason = "timed out" if result.timed_out else f"exit code {result.returncode}"
                return Execution(False, f"{plan.name}: {reason}"), run_dirs
            if links := symlinks_in(out):
                return Execution(
                    False, f"{plan.name}: symbolic link in outputs: {links[0]}"
                ), run_dirs
        return Execution(True), run_dirs

    # -- fixed lifecycle -------------------------------------------------------

    def evaluate_outputs(
        self, output_dir: Path, case: Any, run_dirs: dict[str, Path] | None = None
    ) -> tuple[CheckResult, ...]:
        """Grade outputs already on disk (used directly by negative-control tests)."""
        ctx = CheckContext(output_dir=output_dir, truth=case, run_dirs=run_dirs or {})
        return tuple(self.checks().evaluate(ctx))

    def grade_case(
        self, submission: Path, seed: int, runner: Runner, workdir: Path, lang: str = "python"
    ) -> CaseResult:
        case = self.make_case(seed)
        case_dir = workdir / f"case-{to_hex(seed)[:12]}"
        input_dir = case_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        case_dir.chmod(0o755)
        self.write_inputs(case, input_dir)
        staged = stage_submission(submission, case_dir / "submission")
        execution, run_dirs = self.execute(case, staged, runner, case_dir, lang)
        if not execution.ok:
            skipped = tuple(
                CheckResult(code, Status.SKIPPED, f"skipped: {RUN_CODE} failed")
                for code in self.checks().codes()
            )
            failed = CheckResult(RUN_CODE, Status.FAIL, execution.reason)
            return CaseResult(seed, (failed, *skipped))
        main_dir = run_dirs.get(MAIN, case_dir / "output")
        results = self.evaluate_outputs(main_dir, case, run_dirs)
        return CaseResult(seed, (CheckResult(RUN_CODE, Status.PASS), *results))

    def grade(
        self,
        submission: Path,
        seeds: Sequence[int],
        runner: Runner,
        workdir: Path,
        lang: str = "python",
    ) -> Verdict:
        if lang not in self.languages():
            raise ValueError(f"{self.task_id} accepts {self.languages()}, not {lang!r}")
        cases = tuple(self.grade_case(submission, s, runner, workdir, lang) for s in seeds)
        return Verdict(self.task_id, self.version, cases)
