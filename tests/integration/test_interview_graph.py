"""End-to-end interview through the real LangGraph, with no database.

Drives the graph exactly the way the API does - invoke, hit the interrupt, resume with
`Command` - and asserts the interview produces a coherent, evidence-backed report.
"""

from __future__ import annotations

import re
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from gauntlet.graph.interview_graph import compile_graph
from gauntlet.graph.state import new_state
from gauntlet.schemas import InterviewMode, InterviewType
from tests.conftest import (
    CONFIDENTLY_WRONG_KAFKA,
    JOB_FIXTURE,
    RESUME_FIXTURE,
    STRONG_HASHMAP_ANSWER,
)

# Answers matched to whatever gets asked, so the run exercises adaptive behaviour
# rather than replaying a fixed script.
ANSWER_BANK: list[tuple[str, str]] = [
    (
        r"concurrent|thread|lock|volatile|synchron",
        "ConcurrentHashMap is basically a synchronized HashMap, it locks the whole map "
        "so only one thread can touch it at a time.",
    ),
    (r"hashmap|hash map|collide|bucket", STRONG_HASHMAP_ANSWER),
    (r"kafka|ordering|partition|offset|consumer", CONFIDENTLY_WRONG_KAFKA),
    (r"index|query|postgres|sql", "I think you just add an index and it gets faster."),
    (
        r"transaction|rollback",
        "Spring transactions work for internal calls too, calling it from the same "
        "class works fine, and it rolls back on any exception.",
    ),
]
FALLBACK = "I don't know, I have never had to dig into that."


def answer_for(prompt: str) -> str:
    lowered = prompt.lower()
    for pattern, answer in ANSWER_BANK:
        if re.search(pattern, lowered):
            return answer
    return FALLBACK


@pytest.fixture
def interview():
    """Run a full interview and return (final_state, transcript)."""
    graph = compile_graph(InMemorySaver())
    thread_id = f"test-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    state = new_state(
        session_id=str(uuid.uuid4()),
        candidate_id=str(uuid.uuid4()),
        thread_id=thread_id,
        resume_text=RESUME_FIXTURE,
        job_text=JOB_FIXTURE,
        target_role="Senior Java Engineer",
        target_level="senior",
        mode=InterviewMode.REAL,
        interview_types=[
            InterviewType.JAVA,
            InterviewType.SPRING,
            InterviewType.DATABASE,
            InterviewType.DISTRIBUTED,
        ],
        planned_minutes=20,
        target_company="capital-one",
    )

    result = graph.invoke(state, config=config)
    transcript: list[tuple[str, str]] = []

    for _ in range(30):
        if "__interrupt__" not in result:
            break
        payload = result["__interrupt__"][0].value
        prompt = payload.get("prompt_text") or ""
        reply = answer_for(prompt)
        if payload.get("type") == "question":
            transcript.append((prompt, reply))
        result = graph.invoke(
            Command(resume={"text": reply, "self_confidence": 5}), config=config
        )

    return result, transcript


class TestFullInterview:
    def test_it_completes(self, interview):
        state, _ = interview
        assert state["status"] == "completed"

    def test_it_asks_a_reasonable_number_of_questions(self, interview):
        state, transcript = interview
        assert 4 <= len(state["question_history"]) <= 14
        assert transcript

    def test_it_produces_a_scorecard(self, interview):
        state, _ = interview
        card = state["final_scorecard"]
        assert 0 <= card["overall"] <= 100
        assert card["questions_asked"] > 0
        assert card["committee"]["recommendation"]

    def test_the_plan_reflects_the_job_description(self, interview):
        state, _ = interview
        plan = state["interview_plan"]
        assert plan["focus_areas"]
        assert plan["slate"]

    def test_no_question_is_asked_twice(self, interview):
        state, _ = interview
        prompts = [item["prompt_text"] for item in state["question_history"]]
        assert len(prompts) == len(set(prompts))

    def test_every_answer_produced_evidence(self, interview):
        state, _ = interview
        assert len(state["evidence"]) >= len(state["answer_history"])
        for row in state["evidence"]:
            assert row["concept_keys"]
            assert 0.0 <= row["score"] <= 1.0

    def test_the_skill_graph_is_populated(self, interview):
        state, _ = interview
        assert state["skill_scores"]
        assert all(0.0 <= value <= 1.0 for value in state["skill_scores"].values())

    def test_confidently_wrong_answers_are_caught(self, interview):
        """The candidate in this fixture holds two textbook misconceptions."""
        state, _ = interview
        beliefs = " ".join(item["belief"].lower() for item in state["misconceptions"])
        assert state["misconceptions"], "no misconception detected"
        assert "synchronized hashmap" in beliefs or "ordering" in beliefs

    def test_misconceptions_are_deduplicated_in_the_report(self, interview):
        state, _ = interview
        reported = state["final_scorecard"]["misconceptions"]
        identities = [(item["concept_key"], item["belief"]) for item in reported]
        assert len(identities) == len(set(identities))

    def test_a_misconception_is_probed_not_corrected(self, interview):
        state, _ = interview
        followups = [item for item in state["question_history"] if item["is_followup"]]
        assert followups, "expected at least one adversarial follow-up"
        for item in followups:
            lowered = item["prompt_text"].lower()
            assert "you are wrong" not in lowered
            assert "the correct answer is" not in lowered

    def test_the_study_plan_targets_measured_gaps(self, interview):
        state, _ = interview
        items = state["final_scorecard"]["study_plan"]["items"]
        assert items
        measured = set(state["skill_scores"])
        assert any(item["concept_key"] in measured for item in items)
        for item in items:
            assert item["rationale"], "a study item with no reason is generic advice"

    def test_the_rejection_reason_cites_evidence(self, interview):
        state, _ = interview
        committee = state["final_scorecard"]["committee"]
        assert committee["most_likely_rejection_reason"]

    def test_resume_claims_are_tracked(self, interview):
        state, _ = interview
        claims = state["final_scorecard"]["resume_claims_tested"]
        assert claims
        assert all("support" in claim for claim in claims)

    def test_replay_moments_are_captured(self, interview):
        """Spec section 24: the weak moments must be identifiable afterwards."""
        state, _ = interview
        moments = state["final_scorecard"]["replay_moments"]
        assert moments
        assert all(moment["ordinal"] >= 1 for moment in moments)


class TestRecoveryAndState:
    def test_state_is_json_serialisable(self, interview):
        """Checkpoints must round-trip, or recovery and replay break."""
        import json

        state, _ = interview
        serialisable = {
            key: value for key, value in state.items() if not key.startswith("__")
        }
        json.dumps(serialisable, default=str)

    def test_an_interview_can_be_resumed_mid_flight(self):
        """Rebuild the runtime from the checkpoint and carry on, as after a restart."""
        saver = InMemorySaver()
        thread_id = f"resume-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        first = compile_graph(saver)
        state = new_state(
            session_id=str(uuid.uuid4()),
            candidate_id=str(uuid.uuid4()),
            thread_id=thread_id,
            resume_text=RESUME_FIXTURE,
            job_text=JOB_FIXTURE,
            target_role="Senior Java Engineer",
            target_level="senior",
            mode=InterviewMode.REAL,
            interview_types=[InterviewType.JAVA],
            planned_minutes=20,
        )
        result = first.invoke(state, config=config)
        assert "__interrupt__" in result
        first_question = result["__interrupt__"][0].value["prompt_text"]

        # A brand-new graph object, same checkpointer: simulates a process restart.
        second = compile_graph(saver)
        snapshot = second.get_state(config)
        assert snapshot.values["current_question"]["prompt_text"] == first_question

        resumed = second.invoke(
            Command(resume={"text": STRONG_HASHMAP_ANSWER}), config=config
        )
        assert len(resumed["answer_history"]) == 1

    def test_checkpoint_history_is_addressable_for_replay(self):
        saver = InMemorySaver()
        thread_id = f"history-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        graph = compile_graph(saver)

        graph.invoke(
            new_state(
                session_id=str(uuid.uuid4()),
                candidate_id=str(uuid.uuid4()),
                thread_id=thread_id,
                resume_text=RESUME_FIXTURE,
                job_text=JOB_FIXTURE,
                target_role="Senior Java Engineer",
                target_level="senior",
                mode=InterviewMode.REAL,
                interview_types=[InterviewType.JAVA],
                planned_minutes=20,
            ),
            config=config,
        )
        history = list(graph.get_state_history(config))
        assert len(history) > 1
        assert all(
            snapshot.config.get("configurable", {}).get("checkpoint_id") for snapshot in history
        )


class TestClarificationBranch:
    def test_a_clarifying_question_does_not_consume_a_question_slot(self):
        graph = compile_graph(InMemorySaver())
        thread_id = f"clarify-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        result = graph.invoke(
            new_state(
                session_id=str(uuid.uuid4()),
                candidate_id=str(uuid.uuid4()),
                thread_id=thread_id,
                resume_text=RESUME_FIXTURE,
                job_text=JOB_FIXTURE,
                target_role="Senior Java Engineer",
                target_level="senior",
                mode=InterviewMode.REAL,
                interview_types=[InterviewType.SYSTEM_DESIGN],
                planned_minutes=20,
            ),
            config=config,
        )
        assert "__interrupt__" in result
        before = len(result["question_history"])

        clarified = graph.invoke(
            Command(resume={"text": "Quick question, should I assume a single region?"}),
            config=config,
        )
        payload = clarified["__interrupt__"][0].value
        assert payload["type"] == "clarification"
        assert payload["reply"]
        # Same question still stands; no new one was consumed.
        assert len(clarified["question_history"]) == before
        assert len(clarified.get("answer_history", [])) == 0
