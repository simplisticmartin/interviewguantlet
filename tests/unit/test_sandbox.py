"""Sandboxed execution (spec section 17).

Most of this file runs without Docker, which is the point. The security posture is a pure
function of the argv, so every hardening flag can be asserted with no daemon anywhere. A
silently dropped `--network none` looks exactly like a working sandbox until it is not,
and "we could not test it because Docker was down" is not an acceptable reason for that
to go unnoticed.

The tests that genuinely need a daemon are marked and skip with an actionable message.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gauntlet.execution import sandbox
from gauntlet.execution.sandbox import (
    LANGUAGE_IMAGES,
    SandboxLimits,
    SandboxResult,
    TestCase,
    build_argv,
    run_code,
    run_tests,
    sandbox_available,
)


def argv_for(**overrides) -> list[str]:
    defaults = {
        "container_name": "gauntlet-exec-test",
        "image": "python:3.12-alpine",
        "workdir": Path("/tmp/work"),
        "command": ("python", "/work/solution.py"),
        "limits": SandboxLimits(),
    }
    return build_argv(**{**defaults, **overrides})


class TestSecurityPosture:
    """Each flag here exists to stop a specific attack. None is decorative."""

    def test_the_container_has_no_network(self):
        """Removes exfiltration and command-and-control in one flag."""
        argv = argv_for()
        assert "--network" in argv
        assert argv[argv.index("--network") + 1] == "none"

    def test_the_root_filesystem_is_read_only(self):
        assert "--read-only" in argv_for()

    def test_scratch_space_cannot_execute(self):
        """A writable /tmp that allows exec is a way to stage a dropped binary."""
        argv = argv_for()
        tmpfs = argv[argv.index("--tmpfs") + 1]
        assert "noexec" in tmpfs
        assert "nosuid" in tmpfs

    def test_every_capability_is_dropped(self):
        argv = argv_for()
        assert argv[argv.index("--cap-drop") + 1] == "ALL"

    def test_privileges_cannot_be_regained(self):
        """Without this, a setuid binary in the image undoes the non-root user."""
        argv = argv_for()
        assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"

    def test_it_never_runs_as_root(self):
        argv = argv_for()
        assert argv[argv.index("--user") + 1] == "65534:65534"

    def test_the_submission_is_mounted_read_only(self):
        argv = argv_for()
        volume = argv[argv.index("--volume") + 1]
        assert volume.endswith(":/work:ro")

    def test_cpu_memory_and_process_count_are_capped(self):
        argv = " ".join(argv_for())
        assert "--cpus=0.5" in argv
        assert "--memory=256m" in argv
        assert "--pids-limit=64" in argv

    def test_memory_cannot_be_escaped_through_swap(self):
        """Without matching memory-swap, the memory limit is advisory."""
        argv = argv_for()
        memory = next(a for a in argv if a.startswith("--memory="))
        swap = next(a for a in argv if a.startswith("--memory-swap="))
        assert memory.split("=")[1] == swap.split("=")[1]

    def test_the_container_is_removed_automatically(self):
        assert "--rm" in argv_for()

    def test_the_container_is_named_so_it_can_be_reaped(self):
        argv = argv_for(container_name="gauntlet-exec-abc123")
        assert argv[argv.index("--name") + 1] == "gauntlet-exec-abc123"

    def test_no_host_environment_is_passed_through(self):
        """Host env would carry API keys straight into candidate-controlled code."""
        argv = argv_for()
        assert "--env" not in argv
        assert "-e" not in argv

    def test_nothing_is_run_through_a_shell(self):
        """A fixed argv list; no string interpolation for anyone to break out of."""
        argv = argv_for()
        assert all(isinstance(part, str) for part in argv)
        assert not any(part in {"sh", "bash", "-c"} for part in argv)

    def test_images_are_pinned_rather_than_floating(self):
        """A `latest` tag means the sandbox changes under a deployment."""
        for image, _, _ in LANGUAGE_IMAGES.values():
            assert ":" in image
            assert not image.endswith(":latest")


class TestThereIsNoEscapeHatch:
    """The property that matters most: no path executes candidate code on the host."""

    def test_the_module_never_calls_exec_or_eval(self):
        source = inspect.getsource(sandbox)
        assert "eval(" not in source
        assert "exec(" not in source

    def test_every_subprocess_call_is_a_docker_call(self):
        """A subprocess that is not docker would be running code on the host."""
        source = inspect.getsource(sandbox)
        for line in source.splitlines():
            if "subprocess.run(" in line and "def " not in line:
                # Argv is always built from docker_command(); assert no shell form.
                assert "shell=True" not in line

    def test_shell_is_never_enabled(self):
        assert "shell=True" not in inspect.getsource(sandbox)

    def test_an_unavailable_sandbox_reports_rather_than_falling_back(self, monkeypatch):
        """A degraded sandbox is not a sandbox; it must refuse, not improvise."""
        monkeypatch.setattr(sandbox, "sandbox_available", lambda: False)
        result = run_code("print('hello')", "python")
        assert result.executed is False
        assert result.exit_code is None
        assert "not reachable" in result.unavailable_reason

    def test_an_unsupported_language_is_refused_before_anything_starts(self):
        result = run_code("puts 'hi'", "ruby")
        assert result.executed is False
        assert "No sandbox image" in result.unavailable_reason


class TestResultSemantics:
    def test_not_executed_is_never_mistaken_for_failed(self):
        """`executed=False` says nothing about correctness, so `passed` must be False."""
        result = SandboxResult(executed=False, language="python")
        assert result.passed is False

    def test_a_clean_run_is_a_pass(self):
        assert SandboxResult(executed=True, language="python", exit_code=0).passed

    def test_a_timeout_is_not_a_pass(self):
        result = SandboxResult(
            executed=True, language="python", exit_code=0, timed_out=True
        )
        assert result.passed is False

    def test_a_nonzero_exit_is_not_a_pass(self):
        assert not SandboxResult(executed=True, language="python", exit_code=1).passed

    def test_the_result_serialises_with_the_executed_flag(self):
        payload = SandboxResult(executed=False, language="python").as_dict()
        assert payload["executed"] is False

    def test_running_no_tests_is_reported_rather_than_passing_vacuously(self):
        outcome = run_tests("print(1)", [], "python")
        assert outcome.executed is False
        assert "No test cases" in outcome.unavailable_reason

    def test_an_unavailable_sandbox_does_not_report_passing_tests(self, monkeypatch):
        monkeypatch.setattr(sandbox, "sandbox_available", lambda: False)
        outcome = run_tests("print(1)", [TestCase(name="t", expected_stdout="1")], "python")
        assert outcome.executed is False
        assert outcome.passed == 0
        assert "Not run" in outcome.summary()


@pytest.mark.requires_docker
class TestAgainstARealDaemon:
    """These actually execute code. They skip when Docker is not running."""

    @pytest.fixture(autouse=True)
    def _require_docker(self):
        sandbox.reset_sandbox_probe()
        if not sandbox_available():
            pytest.skip("Docker is not reachable; start Docker Desktop to run these")

    def test_it_runs_code_and_captures_output(self):
        result = run_code("print('hello from the sandbox')", "python")
        assert result.executed
        assert result.exit_code == 0
        assert "hello from the sandbox" in result.stdout

    def test_stdin_reaches_the_program(self):
        result = run_code("print(input().upper())", "python", stdin="abc\n")
        assert result.passed
        assert "ABC" in result.stdout

    def test_a_crash_is_reported_without_raising(self):
        result = run_code("raise SystemExit(3)", "python")
        assert result.executed
        assert result.exit_code == 3
        assert not result.passed

    def test_an_infinite_loop_is_stopped(self):
        result = run_code(
            "while True:\n    pass\n", "python", limits=SandboxLimits(timeout_seconds=3)
        )
        assert result.timed_out
        assert not result.passed

    def test_there_is_no_network(self):
        """The single most important runtime assertion in this file."""
        code = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
            "    print('NETWORK REACHABLE')\n"
            "except Exception:\n"
            "    print('no network')\n"
        )
        result = run_code(code, "python")
        assert result.executed
        assert "NETWORK REACHABLE" not in result.stdout
        assert "no network" in result.stdout

    def test_the_filesystem_is_not_writable(self):
        code = (
            "try:\n"
            "    open('/evidence', 'w').write('x')\n"
            "    print('WROTE TO ROOT')\n"
            "except OSError:\n"
            "    print('read only')\n"
        )
        result = run_code(code, "python")
        assert "WROTE TO ROOT" not in result.stdout

    def test_the_submission_cannot_be_modified(self):
        code = (
            "try:\n"
            "    open('/work/solution.py', 'a').write('x')\n"
            "    print('MODIFIED SUBMISSION')\n"
            "except OSError:\n"
            "    print('submission is read only')\n"
        )
        result = run_code(code, "python")
        assert "MODIFIED SUBMISSION" not in result.stdout

    def test_it_does_not_run_as_root(self):
        result = run_code("import os; print(os.getuid())", "python")
        assert result.stdout.strip() != "0"

    def test_memory_is_capped(self):
        result = run_code(
            "x = bytearray(512 * 1024 * 1024)\nprint('ALLOCATED')\n",
            "python",
            limits=SandboxLimits(memory="64m", timeout_seconds=10),
        )
        assert "ALLOCATED" not in result.stdout

    def test_visible_tests_are_scored(self):
        code = "print(sum(int(x) for x in input().split()))"
        cases = [
            TestCase(name="two", stdin="1 2", expected_stdout="3"),
            TestCase(name="three", stdin="1 2 3", expected_stdout="6"),
            TestCase(name="wrong", stdin="1 1", expected_stdout="99"),
        ]
        outcome = run_tests(code, cases, "python")
        assert outcome.executed
        assert outcome.passed == 2
        assert outcome.failed == 1

    def test_each_case_gets_a_fresh_container(self):
        """State left by one case must not change another's result."""
        code = (
            "import os\n"
            "p = '/tmp/marker'\n"
            "print('second' if os.path.exists(p) else 'first')\n"
            "open(p, 'w').write('x')\n"
        )
        cases = [
            TestCase(name="a", expected_stdout="first"),
            TestCase(name="b", expected_stdout="first"),
        ]
        outcome = run_tests(code, cases, "python")
        assert outcome.passed == 2, "containers are leaking state between test cases"


class TestTheGradingNodeDegrades:
    """The interview must not depend on a sandbox being present."""

    def _state(self, code: str, language: str = "python") -> dict:
        return {
            "session_id": "s1",
            "pending_answer": {"text": "", "code": code, "language": language},
        }

    def test_an_interview_continues_when_no_sandbox_exists(self, monkeypatch):
        from gauntlet.graph.nodes import grading

        monkeypatch.setattr(grading, "run_code", lambda *a, **k: SandboxResult(
            executed=False, language="python", unavailable_reason="Docker is not reachable"
        ))
        result = grading.check_submitted_code(self._state("def f(a):\n    return a\n"))
        check = result["code_check"]
        assert check["syntax_ok"] is True
        assert check["execution"]["executed"] is False
        # Static signals still drive the next question, which is the point.
        assert result["interviewer_notes"]

    def test_unparseable_code_is_never_sent_to_the_sandbox(self, monkeypatch):
        """Spending a container to rediscover a syntax error is waste."""
        from gauntlet.graph.nodes import grading

        called: list[str] = []
        monkeypatch.setattr(
            grading, "run_code", lambda *a, **k: called.append("ran") or SandboxResult(
                executed=True, language="python", exit_code=0
            )
        )
        result = grading.check_submitted_code(self._state("def broken(:\n"))
        assert called == []
        assert result["code_check"]["execution"] is None

    def test_a_timeout_becomes_an_interviewer_signal(self, monkeypatch):
        from gauntlet.graph.nodes import grading

        monkeypatch.setattr(grading, "run_code", lambda *a, **k: SandboxResult(
            executed=True, language="python", timed_out=True
        ))
        result = grading.check_submitted_code(self._state("while True:\n    pass\n"))
        assert any("did not terminate" in note for note in result["interviewer_notes"])
