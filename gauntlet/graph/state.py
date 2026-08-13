"""Interview graph state (spec section 3).

Everything the interview knows lives here, and everything here is JSON-serialisable.
That constraint is deliberate: the LangGraph checkpointer persists this object after
every node, which is what makes session recovery, time travel, and failure replay
possible. Rich objects (``SkillGraph``, Pydantic models) are rehydrated inside nodes
from their serialised form rather than stored live.

One deviation from the spec sketch: ``difficulty`` is an ``int`` (1-5) rather than a
string. The adaptive router adds and subtracts from it on every turn, and a numeric
scale is the honest representation of that. A human-readable label is derived for the UI.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from gauntlet.schemas import InterviewMode, InterviewType
from gauntlet.skills.graph import SkillGraph
from gauntlet.skills.mastery import Evidence

DIFFICULTY_LABELS = {
    1: "warm-up",
    2: "foundational",
    3: "applied",
    4: "deep",
    5: "expert",
}


class InterviewState(TypedDict, total=False):
    """The full interview state, checkpointed after every node."""

    # --- Identity ---------------------------------------------------------
    session_id: str
    candidate_id: str
    thread_id: str

    # --- Inputs (untrusted text lives here and is always fenced downstream) --
    resume_text: str
    job_text: str
    resume_profile: dict[str, Any]
    job_description: dict[str, Any]

    target_company: str | None
    target_role: str
    target_level: str
    mode: str
    interview_types: list[str]

    # --- Company intelligence --------------------------------------------
    company_patterns: dict[str, Any]

    # --- Plan -------------------------------------------------------------
    interview_plan: dict[str, Any]
    plan_cursor: int

    # --- Live turn --------------------------------------------------------
    pending_target: dict[str, Any] | None
    current_question: dict[str, Any] | None
    pending_answer: dict[str, Any] | None
    pending_clarification: dict[str, Any] | None
    pending_coaching: dict[str, Any] | None
    last_classification: dict[str, Any] | None
    last_evaluation: dict[str, Any] | None
    last_decision: dict[str, Any] | None
    code_check: dict[str, Any] | None

    # --- Accumulating history (append-only via reducers) ------------------
    question_history: Annotated[list[dict[str, Any]], operator.add]
    answer_history: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    misconceptions: Annotated[list[dict[str, Any]], operator.add]
    interviewer_notes: Annotated[list[str], operator.add]

    # --- Derived skill picture -------------------------------------------
    skill_scores: dict[str, float]
    confidence_scores: dict[str, float]

    # --- Clock and difficulty --------------------------------------------
    started_at: str
    planned_minutes: int
    elapsed_time: int
    remaining_time: int
    difficulty: int
    followups_on_concept: int
    current_concept_key: str | None

    # --- Reserved for MCP tool calls (phase 4) ---------------------------
    pending_tool_calls: list[dict[str, Any]]

    # --- Output -----------------------------------------------------------
    final_scorecard: dict[str, Any]
    status: str


def new_state(
    *,
    session_id: str,
    candidate_id: str,
    thread_id: str,
    resume_text: str,
    job_text: str,
    target_role: str,
    target_level: str,
    mode: InterviewMode,
    interview_types: list[InterviewType],
    planned_minutes: int,
    target_company: str | None = None,
) -> InterviewState:
    return InterviewState(
        session_id=session_id,
        candidate_id=candidate_id,
        thread_id=thread_id,
        resume_text=resume_text,
        job_text=job_text,
        resume_profile={},
        job_description={},
        target_company=target_company,
        target_role=target_role,
        target_level=target_level,
        mode=mode.value,
        interview_types=[item.value for item in interview_types],
        company_patterns={},
        interview_plan={},
        plan_cursor=0,
        pending_target=None,
        current_question=None,
        pending_answer=None,
        pending_clarification=None,
        pending_coaching=None,
        last_classification=None,
        last_evaluation=None,
        last_decision=None,
        code_check=None,
        question_history=[],
        answer_history=[],
        evidence=[],
        misconceptions=[],
        interviewer_notes=[],
        skill_scores={},
        confidence_scores={},
        started_at=datetime.now(UTC).isoformat(),
        planned_minutes=planned_minutes,
        elapsed_time=0,
        remaining_time=planned_minutes * 60,
        difficulty=3,
        followups_on_concept=0,
        current_concept_key=None,
        pending_tool_calls=[],
        final_scorecard={},
        status="created",
    )


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def elapsed_seconds(state: InterviewState, now: datetime | None = None) -> int:
    started = state.get("started_at")
    if not started:
        return int(state.get("elapsed_time", 0))
    moment = now or datetime.now(UTC)
    try:
        start = datetime.fromisoformat(started)
    except ValueError:
        return int(state.get("elapsed_time", 0))
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    return max(0, int((moment - start).total_seconds()))


def remaining_seconds(state: InterviewState, now: datetime | None = None) -> int:
    budget = int(state.get("planned_minutes", 20)) * 60
    return max(0, budget - elapsed_seconds(state, now))


def minutes_remaining(state: InterviewState, now: datetime | None = None) -> float:
    return remaining_seconds(state, now) / 60.0


def questions_asked(state: InterviewState) -> int:
    return len(state.get("question_history", []))


def questions_remaining(state: InterviewState) -> int:
    plan = state.get("interview_plan") or {}
    target = int(plan.get("target_question_count", 8))
    return max(0, target - questions_asked(state))


def asked_prompts(state: InterviewState) -> list[str]:
    return [item.get("prompt_text", "") for item in state.get("question_history", [])]


def difficulty_label(difficulty: int) -> str:
    return DIFFICULTY_LABELS.get(max(1, min(5, difficulty)), "applied")


def skill_graph_from_state(state: InterviewState) -> SkillGraph:
    """Rebuild the live skill graph from serialised evidence rows."""
    graph = SkillGraph()
    for row in state.get("evidence", []):
        try:
            observed_at = datetime.fromisoformat(row["observed_at"])
        except (KeyError, ValueError):
            observed_at = datetime.now(UTC)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=UTC)

        item = Evidence(
            score=float(row.get("score", 0.0)),
            difficulty=int(row.get("difficulty", 3)),
            observed_at=observed_at,
            self_confidence=row.get("self_confidence"),
            hints_used=int(row.get("hints_used", 0)),
            is_followup=bool(row.get("is_followup", False)),
            judge_confidence=float(row.get("judge_confidence", 1.0)),
        )
        graph.record(list(row.get("concept_keys", [])), item)

    for finding in state.get("misconceptions", []):
        graph.flag_misconception(finding.get("concept_key"))
    return graph


def mode_of(state: InterviewState) -> InterviewMode:
    try:
        return InterviewMode(state.get("mode", "real"))
    except ValueError:
        return InterviewMode.REAL


def interview_types_of(state: InterviewState) -> list[InterviewType]:
    types: list[InterviewType] = []
    for raw in state.get("interview_types", []):
        try:
            types.append(InterviewType(raw))
        except ValueError:
            continue
    return types
