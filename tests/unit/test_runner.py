import os
import pwd
import tempfile
import time
from pathlib import Path

import pytest

from bioengbench.core.runner import LocalRunner, RunSpec, deterministic_env


def _spec(tmp_path: Path, code: str, **kw) -> RunSpec:
    sub = tmp_path / "sub.py"
    sub.write_text(code)
    (tmp_path / "in").mkdir(exist_ok=True)
    return RunSpec(("{submission}",), sub, tmp_path / "in", tmp_path / "case" / "out", **kw)


def test_success_and_exit_code(tmp_path: Path) -> None:
    assert LocalRunner().run(_spec(tmp_path, "pass\n")).ok
    res = LocalRunner().run(_spec(tmp_path, "raise SystemExit(3)\n"))
    assert res.returncode == 3 and not res.ok


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "child_alive"
    code = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        f"    time.sleep(2); open({str(marker)!r}, 'w').write('x'); os._exit(0)\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    res = LocalRunner().run(_spec(tmp_path, code, timeout_s=0.5))
    assert res.timed_out and time.monotonic() - start < 5
    time.sleep(2.5)
    assert not marker.exists(), "forked child survived the timeout"


def test_memory_limit(tmp_path: Path) -> None:
    res = LocalRunner().run(_spec(tmp_path, "x = bytearray(600 * 1024 * 1024)\n", memory_mb=256))
    assert not res.ok


def test_environment_is_scrubbed_and_single_threaded(tmp_path: Path) -> None:
    env = deterministic_env(tmp_path)
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["PYTHONHASHSEED"] == "0"
    os.environ["SECRET_TOKEN"] = "leak"
    try:
        code = "import os\nraise SystemExit(5 if 'SECRET_TOKEN' in os.environ else 0)\n"
        assert LocalRunner().run(_spec(tmp_path, code)).ok
    finally:
        del os.environ["SECRET_TOKEN"]


@pytest.mark.skipif(os.geteuid() != 0, reason="needs root to drop privileges")
def test_sandbox_user_cannot_read_root_only_files() -> None:
    pwd.getpwnam("nobody")
    tmp_path = Path(tempfile.mkdtemp(dir="/tmp"))  # pytest's tmp root is 0700
    secret_dir = tmp_path / "tests"
    secret_dir.mkdir(mode=0o700)
    (secret_dir / "seeds.txt").write_text("hidden")
    tmp_path.chmod(0o755)
    code = (
        "import sys\n"
        f"try:\n    open({str(secret_dir / 'seeds.txt')!r}).read()\n"
        "except PermissionError:\n    sys.exit(0)\n"
        "sys.exit(9)\n"
    )
    spec = _spec(tmp_path, code)
    spec.submission.chmod(0o644)
    res = LocalRunner(("/usr/bin/python3",), sandbox_user="nobody").run(spec)
    assert res.returncode == 0, res.log_path.read_text()


def test_background_children_die_with_a_clean_exit(tmp_path: Path) -> None:
    """A run that exits 0 must not leave children in its process group."""
    marker = tmp_path / "late"
    code = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        f"    time.sleep(1); open({str(marker)!r}, 'w').write('x'); os._exit(0)\n"
    )
    assert LocalRunner().run(_spec(tmp_path, code)).ok
    time.sleep(1.5)
    assert not marker.exists()


def _has_sandbox() -> bool:
    try:
        pwd.getpwnam("sandbox")
    except KeyError:
        return False
    return os.geteuid() == 0


@pytest.mark.skipif(not _has_sandbox(), reason="needs root and a 'sandbox' user")
def test_detached_daemon_is_reaped_by_user_id() -> None:
    """setsid() + double fork leaves the process group; the runner still
    kills it because it runs as the sandbox user."""
    tmp_path = Path(tempfile.mkdtemp(dir="/tmp"))
    tmp_path.chmod(0o755)
    marker = tmp_path / "case" / "out" / "late"
    code = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    if os.fork() == 0:\n"
        f"        time.sleep(1); open({str(marker)!r}, 'w').write('x')\n"
        "    os._exit(0)\n"
    )
    spec = _spec(tmp_path, code)
    spec.submission.chmod(0o644)
    spec.input_dir.chmod(0o755)
    (tmp_path / "case").mkdir(mode=0o755)
    assert LocalRunner(("/usr/bin/python3",), sandbox_user="sandbox").run(spec).ok
    time.sleep(1.5)
    assert not marker.exists()


@pytest.mark.skipif(not _has_sandbox(), reason="needs root and a 'sandbox' user")
def test_process_cap_stops_a_fork_bomb() -> None:
    tmp_path = Path(tempfile.mkdtemp(dir="/tmp"))
    tmp_path.chmod(0o755)
    code = (
        "import os, sys, time\n"
        "n = 0\n"
        "try:\n"
        "    for _ in range(1000):\n"
        "        if os.fork() == 0:\n"
        "            time.sleep(30); os._exit(0)\n"
        "        n += 1\n"
        "except OSError:\n"
        "    sys.exit(7)\n"
    )
    spec = _spec(tmp_path, code)
    spec.submission.chmod(0o644)
    spec.input_dir.chmod(0o755)
    (tmp_path / "case").mkdir(mode=0o755)
    assert LocalRunner(("/usr/bin/python3",), sandbox_user="sandbox").run(spec).returncode == 7
