"""Verdict data model and reward serialisation.

Everything here is a frozen value object with a deterministic JSON form:
keys are sorted, floats are rounded, and nothing time-dependent is included,
so grading the same submission on the same seeds yields byte-identical output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from bioengbench.core.seeds import to_hex

FLOAT_DIGITS = 9  # rounding for serialised measurements; far below any tolerance


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"  # not evaluated because a gating check failed earlier


@dataclass(frozen=True)
class CheckResult:
    code: str
    status: Status
    message: str = ""
    measured: float | None = None
    threshold: float | None = None

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "status": str(self.status)}
        if self.message:
            out["message"] = self.message
        if self.measured is not None:
            out["measured"] = round(self.measured, FLOAT_DIGITS)
        if self.threshold is not None:
            out["threshold"] = round(self.threshold, FLOAT_DIGITS)
        return out


@dataclass(frozen=True)
class CaseResult:
    seed: int
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)

    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.FAIL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": to_hex(self.seed),
            "passed": self.passed,
            "checks": [r.to_dict() for r in self.results],
        }


@dataclass(frozen=True)
class Verdict:
    task_id: str
    task_version: str
    cases: tuple[CaseResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return bool(self.cases) and all(c.passed for c in self.cases)

    @property
    def check_pass_fraction(self) -> float:
        total = sum(len(c.results) for c in self.cases)
        passed = sum(r.passed for c in self.cases for r in c.results)
        return passed / total if total else 0.0

    def failure_codes(self) -> set[str]:
        return {r.code for c in self.cases for r in c.failures()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_version": self.task_version,
            "passed": self.passed,
            "check_pass_fraction": round(self.check_pass_fraction, FLOAT_DIGITS),
            "cases": [c.to_dict() for c in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def reward(self) -> dict[str, float]:
        """Harbor reward.json: a binary verifiable reward plus diagnostics.

        The training signal is `reward` (all hidden cases must pass every
        check). Partial credit is reported for analysis only, because rewarding
        partially correct physics teaches agents to stop at partially correct.
        """
        return {
            "reward": 1.0 if self.passed else 0.0,
            "check_pass_fraction": round(self.check_pass_fraction, FLOAT_DIGITS),
            "cases_passed": float(sum(c.passed for c in self.cases)),
            "cases_total": float(len(self.cases)),
        }

    def write_reward(self, reward_dir: Path) -> None:
        reward_dir.mkdir(parents=True, exist_ok=True)
        reward = self.reward()
        (reward_dir / "reward.json").write_text(json.dumps(reward, sort_keys=True) + "\n")
        (reward_dir / "reward.txt").write_text(f"{reward['reward']:g}\n")
        (reward_dir / "verdict.json").write_text(self.to_json())
