"""Provenance for every validation result: what was run, on what, by what.

A number in the validation report is only as good as the record of how it
was produced. Each result file carries this block; the report links every
figure to its result file, and the manifest hashes every result file.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def tool_version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (out.stdout or out.stderr).splitlines()
    return lines[0] if lines else None


def collect(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    versions = {}
    for pkg in ("bioengbench", "numpy", "scipy", "pint"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        "agent_requirements_sha256": _sha256(ROOT / "docker/agent-requirements.txt"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "octave": tool_version(["octave-cli", "--version"]),
        "argv": sys.argv,
    }
    record.update(extra or {})
    return record


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
