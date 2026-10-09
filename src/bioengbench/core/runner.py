"""Execute an untrusted submission on one hidden case (hexagonal port + adapters).

Threat model: the submission is adversarial. Inside the verifier it runs in
the same container as the grader, so without isolation it could read the
hidden seeds, rebuild the ground truth with the grader's own generator and
write the clean answer. Defences, in order of strength:

1. Separate verifier container (task.toml) so nothing the agent left behind
   runs before grading.
2. Drop to an unprivileged user that cannot read /tests (grader code, seeds)
   or /opt/grader (the generator).
3. Resource limits (address space, CPU time, file size, wall-clock timeout,
   with the whole process group killed on timeout) and a scrubbed environment
   with single-threaded maths libraries.

Only the return code and a timeout flag enter the verdict. Captured stderr
goes to a log file, because it may contain temporary paths that would break
byte-identical verdicts across runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import pwd
import resource
import select
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol

# Every variable a numerical library might read to choose a thread count.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class RunSpec:
    """What to run, with logical paths the runner maps to real ones.

    `argv` may contain the placeholders {submission}, {input_dir} and
    {output_dir}; each runner substitutes paths valid where the code runs.
    """

    argv: tuple[str, ...]
    submission: Path
    input_dir: Path
    output_dir: Path
    timeout_s: float = 60.0
    memory_mb: int = 2048
    extra_env: tuple[tuple[str, str], ...] = ()

    def env(self, home: Path, submission_dir: str) -> dict[str, str]:
        env = deterministic_env(home)
        env.update({k: v.format(submission_dir=submission_dir) for k, v in self.extra_env})
        return env


@dataclass(frozen=True)
class RunResult:
    returncode: int
    timed_out: bool
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class ProtocolError(RuntimeError):
    """The interactive submission broke the line protocol (crash, timeout, bad JSON)."""


class Session:
    """A running submission that exchanges one JSON object per line.

    Used for closed-loop tasks: the grader keeps the plant (and the truth) in
    its own process and only sends measurements across the boundary.
    `request` enforces a per-step deadline, so a stalled or slow controller
    fails the step instead of hanging grading.
    """

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        log: IO[bytes],
        step_timeout_s: float,
        on_close: Callable[[], None] | None = None,
    ):
        self.proc = proc
        self.log = log
        self.step_timeout_s = step_timeout_s
        self._buffer = b""
        self._on_close = on_close

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(message) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ProtocolError("submission closed its input") from exc
        line = self._readline(time.monotonic() + self.step_timeout_s)
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"not JSON: {line[:80]!r}") from exc
        if not isinstance(reply, dict):
            raise ProtocolError("reply must be a JSON object")
        return reply

    def _readline(self, deadline: float) -> str:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("step deadline exceeded")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise ProtocolError(f"submission exited (code {self.proc.poll()})")
            self._buffer += chunk
            if len(self._buffer) > 1_000_000:
                raise ProtocolError("reply too long")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode(errors="replace")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        if self._on_close is not None:
            self._on_close()
        self.log.close()


class Runner(Protocol):
    def run(self, spec: RunSpec) -> RunResult: ...

    def start(self, spec: RunSpec, step_timeout_s: float) -> Session: ...


def deterministic_env(home: Path) -> dict[str, str]:
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update({var: "1" for var in _THREAD_VARS})
    return env


MAX_FILE_MB = 256  # a submission cannot fill the verifier's disk
MAX_PROCESSES = 128  # per sandbox user


def _rlimits(memory_mb: int, cpu_s: int) -> Callable[[], None]:
    """Resource limits, applied in the child after it has dropped privileges."""

    def apply() -> None:
        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        fsize = MAX_FILE_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        if os.getuid() != 0:  # per-user process cap: stops fork bombs (no effect for root)
            resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))

    return apply


def _run_killing_group(argv: list[str], log: IO[bytes], timeout_s: float, **kw: Any) -> int | None:
    """Run in a new session; on timeout kill the whole process group, so a
    submission cannot leave children running. Returns None on timeout."""
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        **kw,
    )
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return None
    finally:
        # Children left in the group (background jobs) die with the run, even
        # when the main process exited cleanly.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)


def reap_user(uid: int, rounds: int = 20) -> int:
    """Kill every process owned by `uid` (the sandbox user). A submission can
    leave its process group with setsid() and double-fork into a daemon that
    outlives the run; only the user id still identifies it. Returns how many
    processes were killed. Never call this for a uid that real work runs as."""
    if uid == 0:
        raise ValueError("refusing to reap root")
    killed = 0
    for _ in range(rounds):
        victims = []
        for status in Path("/proc").glob("[0-9]*/status"):
            with contextlib.suppress(OSError):
                for line in status.read_text().splitlines():
                    if line.startswith("Uid:"):
                        if int(line.split()[1]) == uid:
                            victims.append(int(status.parent.name))
                        break
        if not victims:
            return killed
        for pid in victims:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
                killed += 1
        time.sleep(0.01)
    return killed


class LocalRunner:
    """Run on this machine. Pass `sandbox_user` when running as root.

    `prefix` is the interpreter command, for example ("/usr/local/bin/python",)
    for Python submissions or ("octave-cli", "--no-gui", "--quiet") for Octave.
    With a sandbox user, the interpreter must be executable by that user.
    """

    def __init__(
        self, prefix: tuple[str, ...] = (sys.executable,), sandbox_user: str | None = None
    ):
        self.prefix = prefix
        self.sandbox_user = sandbox_user

    def _prepare(self, spec: RunSpec) -> tuple[list[str], dict[str, Any]]:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        drop: dict[str, Any] = {}
        if self.sandbox_user is not None:
            entry = pwd.getpwnam(self.sandbox_user)
            os.chown(spec.output_dir, entry.pw_uid, entry.pw_gid)
            drop = {"user": entry.pw_uid, "group": entry.pw_gid, "extra_groups": []}
        argv = [*self.prefix] + [
            a.format(
                submission=spec.submission,
                submission_dir=spec.submission.parent,
                input_dir=spec.input_dir,
                output_dir=spec.output_dir,
            )
            for a in spec.argv
        ]
        kw = {
            "cwd": spec.output_dir,
            "env": spec.env(spec.output_dir, str(spec.submission.parent)),
            "preexec_fn": _rlimits(spec.memory_mb, int(spec.timeout_s) + 5),
            **drop,
        }
        return argv, kw

    def run(self, spec: RunSpec) -> RunResult:
        argv, kw = self._prepare(spec)
        log_path = spec.output_dir.parent / f"{spec.output_dir.name}.log"
        with log_path.open("wb") as log:
            code = _run_killing_group(argv, log, spec.timeout_s, **kw)
        self._reap()
        if code is None:
            return RunResult(returncode=-1, timed_out=True, log_path=log_path)
        return RunResult(returncode=code, timed_out=False, log_path=log_path)

    def start(self, spec: RunSpec, step_timeout_s: float) -> Session:
        argv, kw = self._prepare(spec)
        log = (spec.output_dir.parent / f"{spec.output_dir.name}.log").open("wb")
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
            **kw,
        )
        return Session(proc, log, step_timeout_s, on_close=self._reap)

    def _reap(self) -> None:
        if self.sandbox_user is not None:
            reap_user(pwd.getpwnam(self.sandbox_user).pw_uid)


class DockerRunner:
    """Run inside a throwaway container: no network, read-only inputs, nobody user.

    For grading on a workstation without trusting the submission. Inside a
    Harbor verifier, which is already a container, LocalRunner is used.
    """

    def __init__(self, image: str, prefix: tuple[str, ...] = ("python",)):
        self.image = image
        self.prefix = prefix

    def command(self, spec: RunSpec, interactive: bool = False) -> list[str]:
        argv = [
            a.format(
                submission=f"/sub/{spec.submission.name}",
                submission_dir="/sub",
                input_dir="/in",
                output_dir="/out",
            )
            for a in spec.argv
        ]
        env_flags = [f"--env={k}={v}" for k, v in spec.env(Path("/out"), "/sub").items()]
        return [
            "docker", "run", "--rm", "--network=none", *(["-i"] if interactive else []),
            f"--memory={spec.memory_mb}m", "--cpus=1", "--user=65534:65534",
            "--read-only", "--tmpfs=/tmp",
            f"--volume={spec.submission.parent.resolve()}:/sub:ro",
            f"--volume={spec.input_dir.resolve()}:/in:ro",
            f"--volume={spec.output_dir.resolve()}:/out:rw",
            "--workdir=/out", *env_flags,
            self.image, *self.prefix, *argv,
        ]  # fmt: skip

    def run(self, spec: RunSpec) -> RunResult:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        spec.output_dir.chmod(0o777)
        log_path = spec.output_dir.parent / f"{spec.output_dir.name}.log"
        with log_path.open("wb") as log:
            # 30 s allowance for container start-up; --rm removes it if killed.
            code = _run_killing_group(self.command(spec), log, spec.timeout_s + 30)
        if code is None:
            return RunResult(returncode=-1, timed_out=True, log_path=log_path)
        return RunResult(returncode=code, timed_out=False, log_path=log_path)

    def start(self, spec: RunSpec, step_timeout_s: float) -> Session:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        spec.output_dir.chmod(0o777)
        log = (spec.output_dir.parent / f"{spec.output_dir.name}.log").open("wb")
        proc = subprocess.Popen(
            self.command(spec, interactive=True),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            start_new_session=True,
        )
        # First reply also absorbs container start-up time.
        return Session(proc, log, step_timeout_s + 30)
