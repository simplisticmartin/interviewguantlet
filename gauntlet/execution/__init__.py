"""Candidate code handling.

Today: static analysis only (:mod:`gauntlet.execution.static_check`).
Phase 4: an execution service that compiles and runs submissions inside ephemeral,
network-isolated containers with CPU, memory, and wall-clock caps. Nothing in this
package executes candidate code until that lands.
"""

from gauntlet.execution.static_check import SUPPORTED_LANGUAGES, CodeCheckResult, check_code

__all__ = ["SUPPORTED_LANGUAGES", "CodeCheckResult", "check_code"]
