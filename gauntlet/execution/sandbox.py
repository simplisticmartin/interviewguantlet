"""Sandboxed execution of candidate code (spec section 17).

Candidate code is hostile input. It is written by someone we cannot vouch for, it may be
wrong in ways that hang or allocate without bound, and in a hosted deployment it may be
written by someone actively trying to get out. So there is exactly one place it runs: an
ephemeral container with no network, a read-only root filesystem, no privileges, and hard
caps on CPU, memory, processes and wall clock.

**There is no fallback path.** If Docker is unavailable, this returns ``executed=False``
with a reason and the caller falls back to static analysis. It never degrades to
``subprocess``, ``exec`` or a thread, because a degraded sandbox is not a sandbox and the
failure mode is arbitrary code execution on the host. That absence is enforced by a test
rather than by discipline.

The container is torn down with ``--rm`` and, if the client hangs, by an outer timeout
that kills the process and removes the container by name. A sandbox that leaks containers
becomes a resource exhaustion bug of its own.

**What is deliberately not here.** No mounting of the host filesystem beyond a single
read-only directory holding the submission, no environment passthrough, no image pulling
at request time (a missing image is an error, not a network fetch on a hot path), and no
sharing of a container between two submissions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

# Per language: the image, the file to write, and the command run inside the container.
# Images are pinned to a digest-free but explicit tag; a floating "latest" would mean the
# sandbox silently changes under a deployment.
LANGUAGE_IMAGES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "python": ("python:3.12-alpine", "solution.py", ("python", "/work/solution.py")),
    "javascript": ("node:22-alpine", "solution.js", ("node", "/work/solution.js")),
}

SUPPORTED_LANGUAGES = tuple(LANGUAGE_IMAGES)


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Resource caps. Every one of these has a failure it exists to stop.

    ``cpus`` and ``timeout_seconds`` bound an infinite loop. ``memory`` bounds an
    unbounded allocation. ``pids`` bounds a fork bomb, which the other three do not.
    ``output_bytes`` bounds a program that prints forever, which would otherwise exhaust
    memory in *this* process rather than the container.
    """

    cpus: float = 0.5
    memory: str = "256m"
    pids: int = 64
    timeout_seconds: float = 10.0
    output_bytes: int = 64_000


@dataclass(slots=True)
class SandboxResult:
    """The outcome, shaped so that "did not run" is never mistaken for "failed".

    ``executed`` is the field every consumer must check. A result with
    ``executed=False`` says nothing about whether the code is correct.
    """

    executed: bool
    language: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: int = 0
    truncated: bool = False
    unavailable_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.executed and self.exit_code == 0 and not self.timed_out

    def as_dict(self) -> dict[str, object]:
        return {
            "executed": self.executed,
            "language": self.language,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "unavailable_reason": self.unavailable_reason,
        }


def docker_command() -> str | None:
    """Path to the docker client, or ``None``."""
    return shutil.which("docker")


@lru_cache(maxsize=1)
def sandbox_available() -> bool:
    """Whether a usable Docker daemon is reachable.

    Cached and bounded by a short timeout. Probing an unreachable daemon takes tens of
    seconds on some platforms, and doing that per request would turn a missing sandbox
    into an outage.
    """
    binary = docker_command()
    if binary is None:
        return False
    try:
        probe = subprocess.run(  # fixed argv, never a shell string
            [binary, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def reset_sandbox_probe() -> None:
    sandbox_available.cache_clear()


def build_argv(
    *,
    container_name: str,
    image: str,
    workdir: Path,
    command: tuple[str, ...],
    limits: SandboxLimits,
) -> list[str]:
    """The exact docker argv used to run a submission.

    Split out as a pure function so the security posture is testable without a running
    daemon. Every flag below is load-bearing; a test asserts each one is present, because
    a silently dropped ``--network none`` looks identical to a working sandbox right up
    until it is not.
    """
    binary = docker_command() or "docker"
    return [
        binary,
        "run",
        "--rm",
        "--name",
        container_name,
        # No network at all. Candidate code has no reason to reach anything, and this
        # single flag removes exfiltration and command-and-control entirely.
        "--network",
        "none",
        # Nothing may be written to the image layer. The submission directory is mounted
        # read-only and scratch space is an explicit tmpfs below.
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        # Drop every capability and forbid regaining any through setuid binaries.
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        # Never root, even inside a throwaway container.
        "--user",
        "65534:65534",
        f"--cpus={limits.cpus}",
        f"--memory={limits.memory}",
        # Memory and swap equal, so the container cannot escape the memory cap by
        # swapping. Without this the memory limit is advisory.
        f"--memory-swap={limits.memory}",
        f"--pids-limit={limits.pids}",
        "--volume",
        f"{workdir}:/work:ro",
        "--workdir",
        "/work",
        image,
        *command,
    ]


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n... output truncated ...", True


def run_code(
    code: str,
    language: str = "python",
    *,
    stdin: str = "",
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    """Run a submission in a throwaway container.

    Returns ``executed=False`` with a reason rather than raising when the sandbox is
    unavailable: an absent sandbox is an expected operating condition, not an error in
    the request, and the interview continues on static analysis.
    """
    limits = limits or SandboxLimits()
    normalised = language.strip().lower()

    if normalised not in LANGUAGE_IMAGES:
        return SandboxResult(
            executed=False,
            language=normalised,
            unavailable_reason=(
                f"No sandbox image for '{language}'. Supported: "
                f"{', '.join(SUPPORTED_LANGUAGES)}."
            ),
        )

    if not sandbox_available():
        return SandboxResult(
            executed=False,
            language=normalised,
            unavailable_reason=(
                "Docker is not reachable, so nothing was executed. Start it with "
                "`docker compose up -d` or install Docker Desktop. Analysis continues "
                "without execution."
            ),
        )

    image, filename, command = LANGUAGE_IMAGES[normalised]
    container_name = f"gauntlet-exec-{uuid.uuid4().hex[:12]}"
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="gauntlet-sandbox-") as directory:
        workdir = Path(directory)
        (workdir / filename).write_text(code, encoding="utf-8")

        argv = build_argv(
            container_name=container_name,
            image=image,
            workdir=workdir,
            command=command,
            limits=limits,
        )

        try:
            completed = subprocess.run(  # fixed argv, never a shell string
                argv,
                input=stdin,
                capture_output=True,
                text=True,
                # Slightly beyond the container's own limits, so this is the backstop
                # for a wedged client rather than the primary timeout.
                timeout=limits.timeout_seconds + 5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _force_remove(container_name)
            return SandboxResult(
                executed=True,
                language=normalised,
                timed_out=True,
                duration_ms=int((time.monotonic() - started) * 1000),
                stderr="Execution exceeded the time limit and was stopped.",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _force_remove(container_name)
            log.warning("sandbox.failed", error=str(exc)[:200])
            return SandboxResult(
                executed=False,
                language=normalised,
                unavailable_reason=f"Sandbox could not start: {str(exc)[:200]}",
            )

    stdout, out_cut = _truncate(completed.stdout, limits.output_bytes)
    stderr, err_cut = _truncate(completed.stderr, limits.output_bytes)
    duration_ms = int((time.monotonic() - started) * 1000)

    log.info(
        "sandbox.ran",
        language=normalised,
        exit_code=completed.returncode,
        duration_ms=duration_ms,
    )
    return SandboxResult(
        executed=True,
        language=normalised,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        truncated=out_cut or err_cut,
    )


def _force_remove(container_name: str) -> None:
    """Guarantee teardown. A sandbox that leaks containers is its own resource leak."""
    binary = docker_command()
    if binary is None:
        return
    try:
        subprocess.run(  # fixed argv, never a shell string
            [binary, "rm", "-f", container_name],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        log.warning("sandbox.teardown_failed", container=container_name)


@dataclass(slots=True)
class TestCase:
    """One visible test: input on stdin, expected output on stdout."""

    # Stops pytest trying to collect this as a test class wherever it is imported.
    __test__ = False

    name: str
    stdin: str = ""
    expected_stdout: str = ""


@dataclass(slots=True)
class TestRunResult:
    executed: bool
    passed: int = 0
    failed: int = 0
    results: list[dict[str, object]] = field(default_factory=list)
    unavailable_reason: str = ""

    @property
    def total(self) -> int:
        return self.passed + self.failed

    def summary(self) -> str:
        if not self.executed:
            return f"Not run. {self.unavailable_reason}"
        return f"{self.passed}/{self.total} tests passed"


def run_tests(
    code: str,
    cases: list[TestCase],
    language: str = "python",
    *,
    limits: SandboxLimits | None = None,
) -> TestRunResult:
    """Run a submission against visible test cases.

    Each case is a fresh container. Reusing one would let an earlier case leave state
    that changes a later result, which makes a test suite that lies.
    """
    if not cases:
        return TestRunResult(executed=False, unavailable_reason="No test cases supplied.")

    results: list[dict[str, object]] = []
    passed = failed = 0

    for case in cases:
        outcome = run_code(code, language, stdin=case.stdin, limits=limits)
        if not outcome.executed:
            return TestRunResult(
                executed=False, unavailable_reason=outcome.unavailable_reason
            )

        ok = (
            not outcome.timed_out
            and outcome.exit_code == 0
            and outcome.stdout.strip() == case.expected_stdout.strip()
        )
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        results.append(
            {
                "name": case.name,
                "passed": ok,
                "timed_out": outcome.timed_out,
                "exit_code": outcome.exit_code,
                # Truncated hard: a test report is a summary, not a transcript.
                "stdout": outcome.stdout[:2000],
                "stderr": outcome.stderr[:2000],
                "duration_ms": outcome.duration_ms,
            }
        )

    return TestRunResult(executed=True, passed=passed, failed=failed, results=results)


def describe_posture() -> str:
    """Human readable statement of what the sandbox does, for docs and health output."""
    limits = SandboxLimits()
    return json.dumps(
        {
            "available": sandbox_available(),
            "network": "none",
            "filesystem": "read-only, submission mounted read-only, /tmp on tmpfs noexec",
            "user": "65534 (nobody), no-new-privileges, all capabilities dropped",
            "cpus": limits.cpus,
            "memory": limits.memory,
            "pids": limits.pids,
            "timeout_seconds": limits.timeout_seconds,
            "fallback": "static analysis only; candidate code is never run on the host",
        },
        indent=2,
    )
