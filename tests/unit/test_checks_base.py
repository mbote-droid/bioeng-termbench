from pathlib import Path

from bioengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from bioengbench.core.scoring import Status


@check("A", gate=True)
def gate_fails(ctx: CheckContext) -> Outcome:
    return Outcome(False, "nope")


@check("B")
def passes(ctx: CheckContext) -> Outcome:
    return Outcome(True)


@check("C")
def crashes(ctx: CheckContext) -> Outcome:
    raise ZeroDivisionError("boom")


def _ctx() -> CheckContext:
    return CheckContext(output_dir=Path("."), truth=None)


def test_gate_skips_later_siblings_including_groups() -> None:
    suite = CheckGroup("s", [gate_fails, passes, CheckGroup("g", [passes, crashes])])
    results = suite.evaluate(_ctx())
    assert [r.code for r in results] == ["A", "B", "B", "C"]
    assert results[0].status is Status.FAIL
    assert all(r.status is Status.SKIPPED for r in results[1:])


def test_exception_inside_check_is_a_failure_not_a_crash() -> None:
    (result,) = crashes.evaluate(_ctx())
    assert result.status is Status.FAIL
    assert "ZeroDivisionError" in result.message


def test_codes_are_listed_in_order() -> None:
    suite = CheckGroup("s", [passes, CheckGroup("g", [gate_fails, crashes])])
    assert suite.codes() == ["B", "A", "C"]
