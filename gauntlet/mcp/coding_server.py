"""coding MCP server (spec sections 16 and 17).

Analyses candidate-submitted code and turns its structure into interview signals.

**This server deliberately exposes no execution tools.** The spec's tool list includes
`compile_code`, `run_visible_tests` and `run_hidden_tests`, and none of them are here,
because candidate code is hostile input and the only safe place to run hostile input is
an ephemeral, network-isolated, resource-capped container that does not exist yet.

Shipping a tool named `run_hidden_tests` that secretly only parsed the code would be far
worse than not shipping it: an agent would call it, believe the result, and tell a
candidate their solution passed. So the tools here describe exactly what they do, and
every response carries `executed: false`.

What is here is genuinely useful. A triple-nested loop, a missing empty-input guard or
unbounded recursion are all things a human interviewer notices by reading code, and each
one suggests a specific next question. That needs no sandbox.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from gauntlet.execution.static_check import SUPPORTED_LANGUAGES, check_code

server = MCPServer(
    name="gauntlet-coding",
    title="Gauntlet Coding Analysis",
    version="0.1.0",
    instructions=(
        "Statically analyse candidate code and derive interview follow-up signals. "
        "This server NEVER executes code. No tool here compiles or runs anything, and "
        "every response reports executed=false. Do not tell a candidate their code "
        "passed or failed tests based on these results."
    ),
)


@server.tool(
    description=(
        "Statically analyse submitted code. Returns syntax validity, structure and "
        "interview signals. Does NOT execute the code."
    )
)
def analyze_code(code: str, language: str | None = None) -> dict[str, Any]:
    """Parse and inspect a submission without running it."""
    result = check_code(code, language)
    return {
        **result.as_dict(),
        "warning": (
            "Static analysis only. Nothing was executed, so this cannot tell you whether "
            "the code produces correct output."
        ),
    }


@server.tool(
    description=(
        "Interview follow-up questions suggested by the structure of the code, such as "
        "asking about complexity when loops are nested or about assumptions when there "
        "is no empty-input guard."
    )
)
def get_code_signals(code: str, language: str | None = None) -> dict[str, Any]:
    """Turn code structure into things worth asking about."""
    result = check_code(code, language)
    return {
        "language": result.language,
        "syntax_ok": result.syntax_ok,
        "executed": False,
        "signals": result.interviewer_signals,
        "structure": {
            "functions": result.functions,
            "max_loop_depth": result.max_loop_depth,
            "uses_recursion": result.uses_recursion,
            "has_empty_input_guard": result.has_empty_input_guard,
            "line_count": result.line_count,
        },
    }


@server.tool(description="Languages this server can analyse, and what analysis it performs.")
def supported_languages() -> dict[str, Any]:
    """Capabilities, stated plainly."""
    return {
        "languages": list(SUPPORTED_LANGUAGES),
        "capabilities": {
            "syntax_check": True,
            "structure_extraction": True,
            "interview_signals": True,
            "compilation": False,
            "execution": False,
            "test_running": False,
        },
        "note": (
            "Execution requires sandboxed containers with no network and hard resource "
            "limits. Until that exists, no tool here runs code, and none pretends to."
        ),
    }


@server.resource(
    "gauntlet://coding/capabilities",
    description="What this server can and cannot do.",
    mime_type="application/json",
)
def capabilities() -> dict[str, Any]:
    return supported_languages()


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
