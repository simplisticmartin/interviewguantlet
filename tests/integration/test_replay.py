"""Failure replay (spec section 24).

The graph-level half is tested here without a database: forking a session, restating the
exact question, preserving earlier evidence, and dropping everything after the fork point.
The database-backed half is exercised in the API tests.
"""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from gauntlet.graph.interview_graph import compile_graph
from gauntlet.graph.state import new_state
from gauntlet.schemas import InterviewMode, InterviewType
from gauntlet.services.replay import _truncated_state
from tests.conftest import (
    CONFIDENTLY_WRONG_KAFKA,
    JOB_FIXTURE,
    RESUME_FIXTURE,
    STRONG_HASHMAP_ANSWER,
)


@pytest.fixture(scope="module")
def finished_interview():
    """Run a short interview to completion and return its final state."""
    graph = compile_graph(InMemorySaver())
    thread_id = f"replay-src-{uuid.uuid4().hex[:8]}"
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
            interview_types=[InterviewType.JAVA, InterviewType.DISTRIBUTED],
            planned_minutes=20,
        ),
        config=config,
    )
    for _ in range(30):
        if "__interrupt__" not in result:
            break
        result = graph.invoke(
            Command(resume={"text": CONFIDENTLY_WRONG_KAFKA, "self_confidence": 5}),
            config=config,
        )
    return result


class TestTruncation:
    def _replay_record(self):
        class Stub:
            id = uuid.uuid4()
            candidate_id = uuid.uuid4()
            thread_id = "replay-thread"
            planned_minutes = 20

        return Stub()

    def test_history_is_cut_at_the_fork_point(self, finished_interview):
        target_ordinal = 2
        target = next(
            item
            for item in finished_interview["question_history"]
            if item["ordinal"] == target_ordinal
        )
        seeded = _truncated_state(
            finished_interview, target, target_ordinal, self._replay_record()
        )

        ordinals = [item["ordinal"] for item in seeded["question_history"]]
        assert ordinals == [1, 2], ordinals
        # The answer to the replayed question is gone, since it is being reattempted.
        assert [item["ordinal"] for item in seeded["answer_history"]] == [1]

    def test_earlier_evidence_is_preserved(self, finished_interview):
        """The earlier part of the interview genuinely happened, so it still counts."""
        target_ordinal = 3
        target = next(
            item
            for item in finished_interview["question_history"]
            if item["ordinal"] == target_ordinal
        )
        seeded = _truncated_state(
            finished_interview, target, target_ordinal, self._replay_record()
        )
        assert seeded["evidence"], "evidence from before the fork should survive"
        assert all(item["ordinal"] < target_ordinal for item in seeded["evidence"])

    def test_the_exact_question_is_restated(self, finished_interview):
        target = finished_interview["question_history"][1]
        seeded = _truncated_state(
            finished_interview, target, target["ordinal"], self._replay_record()
        )
        assert seeded["current_question"]["prompt_text"] == target["prompt_text"]

    def test_stale_turn_state_is_cleared(self, finished_interview):
        """A fork must not inherit the previous turn's evaluation or routing decision."""
        target = finished_interview["question_history"][1]
        seeded = _truncated_state(
            finished_interview, target, target["ordinal"], self._replay_record()
        )
        for key in (
            "pending_answer",
            "last_evaluation",
            "last_decision",
            "last_classification",
        ):
            assert seeded[key] is None, key
        assert seeded["final_scorecard"] == {}
        assert seeded["status"] == "awaiting_answer"

    def test_the_plan_carries_over(self, finished_interview):
        """A replay is the same interview, so it keeps the same plan."""
        target = finished_interview["question_history"][1]
        seeded = _truncated_state(
            finished_interview, target, target["ordinal"], self._replay_record()
        )
        assert seeded["interview_plan"] == finished_interview["interview_plan"]
        assert seeded["resume_profile"] == finished_interview["resume_profile"]


class TestForkedGraph:
    def test_a_fork_serves_the_original_question_then_adapts(self, finished_interview):
        """End to end at the graph level: fork, restate, answer, continue."""
        target_ordinal = 2
        target = next(
            item
            for item in finished_interview["question_history"]
            if item["ordinal"] == target_ordinal
        )

        class Stub:
            id = uuid.uuid4()
            candidate_id = uuid.uuid4()
            thread_id = f"replay-{uuid.uuid4().hex[:8]}"
            planned_minutes = 20

        replay_record = Stub()
        seeded = _truncated_state(
            finished_interview, target, target_ordinal, replay_record
        )

        graph = compile_graph(InMemorySaver())
        config = {"configurable": {"thread_id": replay_record.thread_id}}
        graph.update_state(config, seeded, as_node="ask_question")

        first = graph.invoke(None, config=config)
        assert "__interrupt__" in first, "the fork should pause on the restated question"
        payload = first["__interrupt__"][0].value
        assert payload["prompt_text"] == target["prompt_text"], (
            "a replay must restate the identical question, not a regenerated variant"
        )

        # Answer it well this time and confirm the interview continues normally.
        second = graph.invoke(
            Command(resume={"text": STRONG_HASHMAP_ANSWER, "self_confidence": 4}),
            config=config,
        )
        assert second.get("last_evaluation") is not None
        assert len(second["answer_history"]) == len(seeded["answer_history"]) + 1

    def test_the_original_interview_is_untouched(self, finished_interview):
        """Forking must not mutate the source, or the comparison is meaningless."""
        before = len(finished_interview["question_history"])
        target = finished_interview["question_history"][1]

        class Stub:
            id = uuid.uuid4()
            candidate_id = uuid.uuid4()
            thread_id = f"replay-{uuid.uuid4().hex[:8]}"
            planned_minutes = 20

        _truncated_state(finished_interview, target, target["ordinal"], Stub())
        assert len(finished_interview["question_history"]) == before
        assert finished_interview["status"] == "completed"


class TestImprovementMeasurement:
    def test_a_better_replay_answer_scores_higher(self, finished_interview):
        """The point of the feature: the delta has to be real and measurable."""
        from gauntlet.evaluation.engine import EvaluationEngine
        from gauntlet.schemas import AnswerPayload, QuestionSpec

        question = QuestionSpec(
            prompt_text="Walk me through a HashMap put that collides.",
            interview_type=InterviewType.JAVA,
            agent_key="java",
            concept_keys=["java.collections.hashmap"],
            difficulty=3,
            rubric_key="java.collections.hashmap",
        )
        engine = EvaluationEngine()
        weak = engine.evaluate(question, AnswerPayload(text="It hashes the key I think."))
        strong = engine.evaluate(question, AnswerPayload(text=STRONG_HASHMAP_ANSWER))

        assert strong.score > weak.score
        assert round(strong.score - weak.score, 4) > 0.3
