"""Interview modes actually behave differently (spec section 25).

Written after an audit found that COACHING, RAPID_FIRE and FULL_LOOP existed in the enum
and in the UI while doing nothing at all. A mode offered to a user that changes no
behaviour is the product lying, so every mode the UI offers now has a test proving it
does something distinct.
"""

from __future__ import annotations

import re
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from gauntlet.agents.planner import question_budget
from gauntlet.graph.interview_graph import compile_graph
from gauntlet.graph.state import new_state
from gauntlet.schemas import InterviewMode, InterviewType
from tests.conftest import CONFIDENTLY_WRONG_KAFKA, JOB_FIXTURE, RESUME_FIXTURE


def run_interview(mode: InterviewMode, minutes: int = 20, max_turns: int = 40):
    """Drive a whole interview in the given mode and return the final state."""
    graph = compile_graph(InMemorySaver())
    thread_id = f"mode-{uuid.uuid4().hex[:8]}"
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
            mode=mode,
            interview_types=[InterviewType.JAVA, InterviewType.DISTRIBUTED],
            planned_minutes=minutes,
        ),
        config=config,
    )

    coaching_notes: list[dict] = []
    for _ in range(max_turns):
        if "__interrupt__" not in result:
            break
        result = graph.invoke(
            Command(resume={"text": CONFIDENTLY_WRONG_KAFKA, "self_confidence": 5}),
            config=config,
        )
        note = result.get("pending_coaching")
        if note:
            coaching_notes.append(note)

    return result, coaching_notes


class TestRealMode:
    def test_real_mode_never_teaches(self):
        """The core guarantee: measuring must not turn into teaching."""
        state, notes = run_interview(InterviewMode.REAL)
        assert notes == [], "Real Interview Mode produced coaching feedback"
        assert not state.get("pending_coaching")

    def test_real_mode_never_reveals_a_correction_mid_interview(self):
        state, _ = run_interview(InterviewMode.REAL)
        # The candidate holds a Kafka misconception throughout; it must be detected...
        assert state["misconceptions"], "expected the misconception to be detected"
        # ...but never stated back to them in a question.
        corrections = {
            row["correction"].lower() for row in state["misconceptions"] if row.get("correction")
        }
        for question in state["question_history"]:
            prompt = question["prompt_text"].lower()
            for correction in corrections:
                assert correction not in prompt, "a correction leaked into a question"


class TestCoachingMode:
    def test_coaching_mode_teaches_between_questions(self):
        _, notes = run_interview(InterviewMode.COACHING)
        assert notes, "Coaching Mode produced no feedback at all"
        assert all(note["feedback"].strip() for note in notes)

    def test_coaching_corrects_a_confidently_wrong_belief(self):
        _, notes = run_interview(InterviewMode.COACHING)
        corrections = [note.get("key_correction") for note in notes if note.get("key_correction")]
        assert corrections, "Coaching Mode never corrected the misconception"

    def test_coaching_never_states_a_score(self):
        """Coaching is about understanding; a number turns it back into grading."""
        _, notes = run_interview(InterviewMode.COACHING)
        for note in notes:
            blob = " ".join(str(value) for value in note.values() if value)
            assert not re.search(r"\b\d{1,3}\s*%|\bscore\b|\b0\.\d+\b", blob, re.I), blob

    def test_coaching_still_produces_a_scorecard(self):
        state, _ = run_interview(InterviewMode.COACHING)
        assert state["status"] == "completed"
        assert state["final_scorecard"]["overall"] >= 0


class TestRapidFireMode:
    def test_rapid_fire_budgets_far_more_questions(self):
        normal = question_budget(20, InterviewMode.REAL)
        rapid = question_budget(20, InterviewMode.RAPID_FIRE)
        assert rapid > normal * 2

    def test_rapid_fire_asks_more_questions_in_the_same_time(self):
        normal, _ = run_interview(InterviewMode.REAL, minutes=20)
        rapid, _ = run_interview(InterviewMode.RAPID_FIRE, minutes=20)
        assert len(rapid["question_history"]) > len(normal["question_history"])

    def test_rapid_fire_does_not_probe(self):
        """Breadth, not depth: follow-ups defeat the format."""
        state, _ = run_interview(InterviewMode.RAPID_FIRE)
        followups = [item for item in state["question_history"] if item["is_followup"]]
        assert followups == [], "rapid fire produced follow-up probes"

    def test_rapid_fire_still_records_evidence(self):
        state, _ = run_interview(InterviewMode.RAPID_FIRE)
        assert state["evidence"]
        assert state["skill_scores"]


class TestModeScoping:
    @pytest.mark.parametrize(
        "mode",
        [
            InterviewMode.CODING,
            InterviewMode.SYSTEM_DESIGN,
            InterviewMode.BEHAVIORAL,
            InterviewMode.RESUME_DEFENSE,
        ],
    )
    def test_single_topic_modes_lock_the_plan(self, mode: InterviewMode):
        """A mode that pins the interview must not wander into other areas."""
        from gauntlet.agents.planner import _MODE_LOCK

        locked = _MODE_LOCK[mode]
        state, _ = run_interview(mode, minutes=10, max_turns=12)
        areas = {area["interview_type"] for area in state["interview_plan"]["focus_areas"]}
        assert areas == {locked.value}, f"{mode.value} planned {areas}"
