"""Candidate code handling.

Two layers, and the difference matters:

* :mod:`gauntlet.execution.static_check` parses a submission and extracts structural
  signals. It never runs anything, and it always works.
* :mod:`gauntlet.execution.sandbox` runs a submission inside an ephemeral, network-less,
  read-only, resource-capped container. It requires Docker, and when Docker is absent it
  reports that rather than falling back.

There is deliberately no third option. Candidate code is never executed on the host.
"""

from gauntlet.execution.sandbox import (
    SandboxLimits,
    SandboxResult,
    TestCase,
    TestRunResult,
    describe_posture,
    reset_sandbox_probe,
    run_code,
    run_tests,
    sandbox_available,
)
from gauntlet.execution.static_check import (
    SUPPORTED_LANGUAGES,
    CodeCheckResult,
    check_code,
)

__all__ = [
    "SUPPORTED_LANGUAGES",
    "CodeCheckResult",
    "SandboxLimits",
    "SandboxResult",
    "TestCase",
    "TestRunResult",
    "check_code",
    "describe_posture",
    "reset_sandbox_probe",
    "run_code",
    "run_tests",
    "sandbox_available",
]
