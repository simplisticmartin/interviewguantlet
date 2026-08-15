"""MCP servers (spec section 16).

Tools are exercised directly rather than over a transport: the transport is the SDK's
job, the tool bodies are ours. Registration is checked separately so a tool cannot be
silently missing from the manifest.
"""

from __future__ import annotations

import pytest

from gauntlet.mcp import candidate_server, coding_server, question_bank_server


async def _tool_names(server) -> set[str]:
    return {tool.name for tool in await server.list_tools()}


class TestQuestionBankServer:
    async def test_tools_are_registered(self):
        names = await _tool_names(question_bank_server.server)
        assert {
            "search_questions",
            "find_question_family",
            "get_topic_questions",
            "get_company_patterns",
            "get_question_metadata",
            "list_concepts",
        } <= names

    def test_search_returns_relevant_questions(self):
        result = question_bank_server.search_questions("kafka ordering guarantees")
        assert result["count"] > 0
        top = result["questions"][0]
        assert "kafka" in (top["question"] + " ".join(top["topics"])).lower()

    def test_every_result_carries_provenance(self):
        """Nothing leaves this server without saying where it came from."""
        result = question_bank_server.search_questions("hashmap")
        for question in result["questions"]:
            assert question["origin"] == "generated"
            assert question["source_type"] == "gauntlet_authored"
            assert "Not attributed to any company" in question["attribution"]

    def test_filters_are_applied(self):
        result = question_bank_server.search_questions("design", interview_type="dsa")
        assert all(item["interview_type"] == "dsa" for item in result["questions"])

    def test_an_invalid_filter_explains_itself(self):
        result = question_bank_server.search_questions("x", interview_type="nonsense")
        assert "error" in result
        assert "valid" in result

    def test_duplicate_detection_flags_a_reword(self):
        result = question_bank_server.find_question_family(
            "Return the indices of two values that sum to a target.",
            concept_keys=["dsa.arrays", "dsa.hashing"],
        )
        assert result["is_duplicate"]
        assert result["matches"]

    def test_duplicate_detection_accepts_something_new(self):
        result = question_bank_server.find_question_family(
            "How would you design a distributed rate limiter using Redis?",
            concept_keys=["system_design.rate_limiting"],
        )
        assert not result["is_duplicate"]

    def test_topic_questions(self):
        result = question_bank_server.get_topic_questions("kafka.ordering")
        assert result["count"] > 0
        assert result["display_name"] == "Kafka ordering guarantees"

    def test_unknown_concept_is_explained(self):
        assert "error" in question_bank_server.get_topic_questions("not.a.concept")

    def test_company_patterns_are_labelled_as_estimates(self):
        """The disclaimer must survive the trip out through MCP."""
        result = question_bank_server.get_company_patterns("google")
        assert result["evidence"] == "estimated"
        assert "no observed interview reports" in result["disclaimer"].lower()
        assert abs(sum(result["distribution"].values()) - 1.0) < 0.01

    def test_unknown_company_lists_known_ones(self):
        result = question_bank_server.get_company_patterns("not-a-real-company")
        assert "error" in result
        assert result["known"]

    def test_question_metadata_includes_the_rubric_and_misconceptions(self):
        result = question_bank_server.get_question_metadata("kafka-ordering-scope")
        assert result["rubric"]["dimensions"]
        assert result["rubric"]["known_misconceptions"]
        assert all(item["probe"] for item in result["rubric"]["dimensions"])

    def test_corpus_stats_report_duplication(self):
        stats = question_bank_server.corpus_stats()
        assert stats["questions"] == stats["families"], "shipped corpus should be clean"
        assert stats["inflation_ratio"] == 1.0


class TestCodingServer:
    async def test_tools_are_registered(self):
        names = await _tool_names(coding_server.server)
        assert {"analyze_code", "get_code_signals", "supported_languages"} <= names

    async def test_no_execution_tool_is_exposed(self):
        """The whole point: nothing here may look like it runs code."""
        names = await _tool_names(coding_server.server)
        forbidden = {"compile_code", "run_visible_tests", "run_hidden_tests", "execute", "run_code"}
        assert not (names & forbidden), f"an execution tool was exposed: {names & forbidden}"

    def test_analysis_always_reports_that_nothing_ran(self):
        result = coding_server.analyze_code("def f(a):\n    return a[0]\n", "python")
        assert result["executed"] is False
        assert "Nothing was executed" in result["warning"]

    def test_signals_are_derived_from_structure(self):
        code = "def f(a):\n    for i in a:\n        for j in a:\n            print(i, j)\n"
        result = coding_server.get_code_signals(code, "python")
        assert result["structure"]["max_loop_depth"] == 2
        assert any("complexity" in signal for signal in result["signals"])
        assert result["executed"] is False

    def test_broken_code_is_reported_not_raised(self):
        result = coding_server.analyze_code("def broken(:", "python")
        assert result["syntax_ok"] is False
        assert result["errors"]

    def test_capabilities_are_stated_honestly(self):
        caps = coding_server.supported_languages()["capabilities"]
        assert caps["syntax_check"] is True
        assert caps["execution"] is False
        assert caps["test_running"] is False


class TestCandidateServer:
    async def test_tools_are_registered(self):
        names = await _tool_names(candidate_server.server)
        assert {
            "get_resume",
            "get_previous_interviews",
            "get_mastery_graph",
            "get_previous_mistakes",
            "get_skill_history",
        } <= names

    @pytest.mark.parametrize(
        "tool",
        [
            candidate_server.get_resume,
            candidate_server.get_previous_interviews,
            candidate_server.get_mastery_graph,
            candidate_server.get_previous_mistakes,
            candidate_server.get_skill_history,
        ],
    )
    def test_every_tool_requires_a_candidate_id(self, tool):
        """No ambient scope. Handing back the wrong person's history is unacceptable."""
        import inspect

        assert "candidate_id" in inspect.signature(tool).parameters

    def test_a_malformed_id_is_rejected_rather_than_guessed(self, db_available: bool):
        if not db_available:
            pytest.skip("needs a database to reach the lookup path")
        assert "error" in candidate_server.get_resume("not-a-uuid")

    def test_database_outage_is_reported_clearly(self, db_available: bool):
        """An unavailable skill graph and an empty one mean different things."""
        if db_available:
            pytest.skip("database is up, so this path cannot be exercised")
        result = candidate_server.get_mastery_graph("00000000-0000-0000-0000-000000000000")
        assert result["error"] == "Database unavailable."
        assert "docker compose" in result["hint"]
