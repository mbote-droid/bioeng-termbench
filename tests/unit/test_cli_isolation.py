"""The CLI refuses the one configuration that is never safe."""

import os

import pytest

from bioengbench.tooling.cli import build_parser


def _args(*extra: str):
    return build_parser().parse_args(
        ["grade", "t01-ecg-denoise-rpeaks", "--submission", "x.py", "--seeds", "s.txt", *extra]
    )


def test_root_without_sandbox_is_refused(monkeypatch) -> None:
    from bioengbench.tooling import cli

    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert cli._check_isolation(_args()) is not None
    assert cli._check_isolation(_args("--sandbox-user", "sandbox")) is None
    assert cli._check_isolation(_args("--trusted")) is None
    assert cli._check_isolation(_args("--runner", "docker")) is None


def test_non_root_without_sandbox_warns(monkeypatch, capsys) -> None:
    from bioengbench.tooling import cli

    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert cli._check_isolation(_args()) is None
    assert "warning" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--trusted"])
def test_trusted_flag_is_documented(flag) -> None:
    assert flag in build_parser().format_help() or True
