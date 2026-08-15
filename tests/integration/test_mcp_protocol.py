"""MCP servers over the real protocol (spec section 16).

The unit tests call tool functions directly, which proves the logic but not that the
servers actually speak MCP. These spawn each server as a subprocess and talk to it with
the SDK client, exactly as Claude Desktop or an IDE would.

Slower than the rest of the suite because each case starts a process, which is why the
tool bodies are covered separately and this file only checks the protocol boundary.
"""

from __future__ import annotations

import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call(module: str, tool: str, arguments: dict) -> dict:
    """Spawn a server, call one tool, return the decoded payload."""
    params = StdioServerParameters(command=sys.executable, args=["-m", module], env=None)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool, arguments)
        assert result.content, f"{tool} returned no content"
        return json.loads(result.content[0].text)


async def tool_names(module: str) -> set[str]:
    params = StdioServerParameters(command=sys.executable, args=["-m", module], env=None)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listing = await session.list_tools()
        return {tool.name for tool in listing.tools}


QUESTION_BANK = "gauntlet.mcp.question_bank_server"
CODING = "gauntlet.mcp.coding_server"
CANDIDATE = "gauntlet.mcp.candidate_server"


class TestProtocolHandshake:
    async def test_question_bank_advertises_its_tools(self):
        names = await tool_names(QUESTION_BANK)
        assert "search_questions" in names
        assert "get_company_patterns" in names

    async def test_coding_server_advertises_no_execution_tool(self):
        """Checked over the wire, since the manifest is what a client actually sees."""
        names = await tool_names(CODING)
        assert {"analyze_code", "get_code_signals"} <= names
        assert not (
            names & {"compile_code", "run_visible_tests", "run_hidden_tests", "run_code"}
        )

    async def test_candidate_server_starts_without_a_database(self):
        """It must advertise its tools even when Postgres is unreachable."""
        names = await tool_names(CANDIDATE)
        assert "get_mastery_graph" in names


class TestToolCallsOverTheWire:
    async def test_search_returns_questions(self):
        payload = await call(QUESTION_BANK, "search_questions", {"query": "kafka ordering"})
        assert payload["count"] > 0
        assert payload["questions"][0]["question"]

    async def test_provenance_survives_the_protocol_boundary(self):
        payload = await call(QUESTION_BANK, "search_questions", {"query": "hashmap"})
        for question in payload["questions"]:
            assert question["source_type"] == "gauntlet_authored"

    async def test_the_company_estimate_disclaimer_survives(self):
        """The honesty constraint has to hold for MCP clients too, not just our own UI."""
        payload = await call(QUESTION_BANK, "get_company_patterns", {"company": "google"})
        assert payload["evidence"] == "estimated"
        assert "no observed interview reports" in payload["disclaimer"].lower()

    async def test_code_analysis_reports_that_nothing_ran(self):
        payload = await call(
            CODING,
            "get_code_signals",
            {
                "code": "def f(a):\n    for i in a:\n        for j in a:\n            print(i)\n",
                "language": "python",
            },
        )
        assert payload["executed"] is False
        assert payload["structure"]["max_loop_depth"] == 2
        assert payload["signals"]

    async def test_capabilities_do_not_overstate(self):
        payload = await call(CODING, "supported_languages", {})
        assert payload["capabilities"]["execution"] is False
        assert payload["capabilities"]["test_running"] is False

    async def test_candidate_tools_report_a_database_outage(self, db_available: bool):
        if db_available:
            pytest.skip("database is up, so the outage path cannot be exercised")
        payload = await call(
            CANDIDATE,
            "get_mastery_graph",
            {"candidate_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert payload["error"] == "Database unavailable."
