"""Check abstractions.

A *check* is a pure function of a CheckContext: same outputs and same truth,
same result. Leaf checks are wrapped functions (Strategy); a CheckGroup
holds checks or groups (Composite). A gating group stops evaluation of later
siblings when it fails, because a missing file or a malformed table would
make every downstream measurement meaningless; those are reported SKIPPED.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bioengbench.core.scoring import CheckResult, Status


@dataclass
class CheckContext:
    """Inputs to every check: where the submission wrote, and the hidden truth."""

    output_dir: Path
    truth: Any
    run_dirs: dict[str, Path] = field(default_factory=dict)  # one per RunPlan
    outputs: dict[str, Any] = field(default_factory=dict)  # filled by loading checks


@dataclass(frozen=True)
class Outcome:
    passed: bool
    message: str = ""
    measured: float | None = None
    threshold: float | None = None


class Check(Protocol):
    @property
    def gate(self) -> bool: ...

    def codes(self) -> list[str]: ...

    def evaluate(self, ctx: CheckContext) -> list[CheckResult]: ...


@dataclass(frozen=True)
class FunctionCheck:
    code: str
    fn: Callable[[CheckContext], Outcome]
    gate: bool = False

    def codes(self) -> list[str]:
        return [self.code]

    def evaluate(self, ctx: CheckContext) -> list[CheckResult]:
        try:
            out = self.fn(ctx)
        except Exception as exc:  # a malformed submission must fail, never crash grading
            return [CheckResult(self.code, Status.FAIL, f"{type(exc).__name__}: {exc}")]
        status = Status.PASS if out.passed else Status.FAIL
        message = "" if out.passed else out.message  # messages describe failures only
        return [CheckResult(self.code, status, message, out.measured, out.threshold)]


def check(
    code: str, *, gate: bool = False
) -> Callable[[Callable[[CheckContext], Outcome]], FunctionCheck]:
    """Decorator: turn `fn(ctx) -> Outcome` into a named check."""

    def wrap(fn: Callable[[CheckContext], Outcome]) -> FunctionCheck:
        return FunctionCheck(code, fn, gate)

    return wrap


@dataclass(frozen=True)
class CheckGroup:
    name: str
    children: Sequence[Check]
    gate: bool = False

    def codes(self) -> list[str]:
        return [code for child in self.children for code in child.codes()]

    def evaluate(self, ctx: CheckContext) -> list[CheckResult]:
        results: list[CheckResult] = []
        blocked_by: str | None = None
        for child in self.children:
            if blocked_by is not None:
                results += [
                    CheckResult(code, Status.SKIPPED, f"skipped: {blocked_by} failed")
                    for code in child.codes()
                ]
                continue
            child_results = child.evaluate(ctx)
            results += child_results
            failed = [r.code for r in child_results if not r.passed]
            if child.gate and failed:
                blocked_by = failed[0]
        return results
