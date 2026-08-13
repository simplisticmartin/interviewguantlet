"""Interview lifecycle: drive the graph, persist everything it produces.

The graph is the source of truth *during* an interview (its checkpoint is what makes
recovery and replay possible). This module mirrors that state into normalised tables
after every turn, so analytics, history, and the cross-interview skill graph can be
queried with SQL rather than by replaying checkpoints.

Sync is idempotent and keyed on (session_id, ordinal), so a retried request cannot
double-insert questions or answers.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session

from gauntlet.config import get_settings
from gauntlet.content.companies import find_company
from gauntlet.db.models import (
    CandidateAnswer,
    Company,
    Evaluation,
    InterviewQuestion,
    InterviewSession,
    JobDescription,
    Misconception,
    Resume,
    StudyPlan,
    StudyPlanItem,
)
from gauntlet.graph.state import InterviewState, new_state
from gauntlet.schemas import AnswerPayload, InterviewMode, InterviewType
from gauntlet.services.runtime import RUNTIME
from gauntlet.services.skills import merge_session_into_skill_graph

log = structlog.get_logger(__name__)


class InterviewError(Exception):
    """The interview could not be advanced."""


@dataclass(frozen=True, slots=True)
class StartRequest:
    resume_id: uuid.UUID | None
    job_description_id: uuid.UUID | None
    target_role: str
    target_level: str
    mode: InterviewMode
    interview_types: list[InterviewType]
    minutes: int
    company_slug: str | None = None


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What the API returns after starting an interview or submitting an answer."""

    session_id: uuid.UUID
    status: str
    question: dict[str, Any] | None
    clarification: dict[str, Any] | None
    coaching: dict[str, Any] | None
    scorecard: dict[str, Any] | None
    remaining_seconds: int
    questions_asked: int


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def start_interview(
    session: Session, candidate_id: uuid.UUID, request: StartRequest
) -> TurnResult:
    settings = get_settings()

    resume_text = ""
    if request.resume_id is not None:
        resume = session.get(Resume, request.resume_id)
        if resume is None or resume.candidate_id != candidate_id:
            raise InterviewError("Resume not found.")
        resume_text = resume.raw_text

    job_text = ""
    if request.job_description_id is not None:
        job = session.get(JobDescription, request.job_description_id)
        if job is None or job.candidate_id != candidate_id:
            raise InterviewError("Job description not found.")
        job_text = job.raw_text

    company_row: Company | None = None
    company_slug: str | None = None
    if request.company_slug:
        seed = find_company(request.company_slug)
        if seed is not None:
            company_slug = seed.slug
            company_row = session.scalar(select(Company).where(Company.slug == seed.slug))

    minutes = max(5, min(120, request.minutes or settings.default_interview_minutes))
    record = InterviewSession(
        candidate_id=candidate_id,
        resume_id=request.resume_id,
        job_description_id=request.job_description_id,
        company_id=company_row.id if company_row else None,
        target_role=request.target_role,
        target_level=request.target_level,
        mode=request.mode.value,
        interview_types=[item.value for item in request.interview_types],
        planned_minutes=minutes,
        status="created",
        thread_id=f"interview-{uuid.uuid4().hex}",
        started_at=datetime.now(UTC),
    )
    session.add(record)
    session.flush()

    state = new_state(
        session_id=str(record.id),
        candidate_id=str(candidate_id),
        thread_id=record.thread_id,
        resume_text=resume_text,
        job_text=job_text,
        target_role=request.target_role,
        target_level=request.target_level,
        mode=request.mode,
        interview_types=request.interview_types,
        planned_minutes=minutes,
        target_company=company_slug,
    )

    log.info(
        "interview.start",
        session=str(record.id),
        role=request.target_role,
        level=request.target_level,
        mode=request.mode.value,
        minutes=minutes,
    )

    result = RUNTIME.graph.invoke(state, config=_config(record.thread_id))
    return _sync_and_build(session, record, result)


def submit_answer(
    session: Session, record: InterviewSession, payload: AnswerPayload
) -> TurnResult:
    if record.status == "completed":
        raise InterviewError("This interview is already complete.")

    resume_value = payload.model_dump(mode="json")
    result = RUNTIME.graph.invoke(Command(resume=resume_value), config=_config(record.thread_id))
    return _sync_and_build(session, record, result)


def finish_interview(session: Session, record: InterviewSession) -> TurnResult:
    """End early and produce the report from the evidence gathered so far."""
    if record.status == "completed":
        return _build_result(record, current_state(record), None)

    # Update the checkpointed state so the report node stops rather than asking more.
    RUNTIME.graph.update_state(
        _config(record.thread_id),
        {"planned_minutes": 0, "remaining_time": 0},
    )
    state = current_state(record)
    from gauntlet.graph.nodes.reporting import build_report

    updates = build_report(state)
    RUNTIME.graph.update_state(_config(record.thread_id), updates)

    merged = {**state, **updates}
    _sync_session(session, record, merged)
    return _build_result(record, merged, None)


def current_state(record: InterviewSession) -> InterviewState:
    snapshot = RUNTIME.graph.get_state(_config(record.thread_id))
    return snapshot.values if snapshot else InterviewState()


def checkpoint_history(record: InterviewSession, limit: int = 50) -> list[dict[str, Any]]:
    """Checkpoint ids, newest first - the anchors failure replay will fork from."""
    history: list[dict[str, Any]] = []
    for snapshot in RUNTIME.graph.get_state_history(_config(record.thread_id), limit=limit):
        values = snapshot.values or {}
        question = values.get("current_question") or {}
        history.append(
            {
                "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id"),
                "next_nodes": list(snapshot.next),
                "ordinal": question.get("ordinal"),
                "prompt_text": question.get("prompt_text"),
                "questions_asked": len(values.get("question_history", [])),
            }
        )
    return history


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _sync_and_build(
    session: Session, record: InterviewSession, result: dict[str, Any]
) -> TurnResult:
    interrupt_payload = _interrupt_payload(result)
    _sync_session(session, record, result)
    return _build_result(record, result, interrupt_payload)


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    value = interrupts[0].value
    return value if isinstance(value, dict) else None


def _sync_session(session: Session, record: InterviewSession, state: Mapping[str, Any]) -> None:
    """Mirror graph state into normalised tables. Idempotent."""
    record.status = str(state.get("status", record.status))
    record.plan = state.get("interview_plan") or record.plan

    existing = {row.ordinal: row for row in record.questions}
    answers_by_ordinal = {
        item.get("ordinal"): item for item in state.get("answer_history", [])
    }
    evaluation = state.get("last_evaluation") or {}
    latest_ordinal = max(answers_by_ordinal) if answers_by_ordinal else None

    for question in state.get("question_history", []):
        ordinal = question.get("ordinal")
        if ordinal is None:
            continue
        row = existing.get(ordinal)
        if row is None:
            row = InterviewQuestion(
                session_id=record.id,
                ordinal=int(ordinal),
                prompt_text=str(question.get("prompt_text", "")),
                interview_type=str(question.get("interview_type", "java")),
                agent_key=str(question.get("agent_key", "java")),
                concept_keys=list(question.get("concept_keys", [])),
                rubric_key=question.get("rubric_key"),
                difficulty=int(question.get("difficulty", 3)),
                is_followup=bool(question.get("is_followup", False)),
                probe_reason=question.get("probe_reason"),
                asks_confidence=bool(question.get("asks_confidence", False)),
                asked_at=_parse_dt(question.get("asked_at")),
            )
            session.add(row)
            session.flush()
            existing[int(ordinal)] = row

        answer = answers_by_ordinal.get(ordinal)
        if answer is None or row.answer is not None:
            continue

        answer_row = CandidateAnswer(
            interview_question_id=row.id,
            text=str(answer.get("text", "")),
            code=answer.get("code"),
            language=answer.get("language"),
            self_confidence=answer.get("self_confidence"),
            latency_ms=answer.get("latency_ms"),
        )
        session.add(answer_row)
        session.flush()

        # Judge verdicts are only carried for the most recent answer in graph state;
        # earlier ones were already persisted on their own turn.
        if ordinal == latest_ordinal and evaluation:
            _persist_evaluations(session, answer_row, evaluation)

    _persist_misconceptions(session, record, state)

    if record.status == "completed" and not record.final_scorecard:
        scorecard = state.get("final_scorecard") or {}
        record.final_scorecard = scorecard
        record.ended_at = datetime.now(UTC)
        merge_session_into_skill_graph(session, record, state)
        _persist_study_plan(session, record, scorecard)
        log.info(
            "interview.completed",
            session=str(record.id),
            overall=scorecard.get("overall"),
            questions=scorecard.get("questions_asked"),
        )


def _persist_evaluations(
    session: Session, answer_row: CandidateAnswer, evaluation: dict[str, Any]
) -> None:
    verdicts = evaluation.get("verdicts") or []
    if not verdicts:
        verdicts = [
            {
                "judge_key": "aggregate",
                "score": evaluation.get("score", 0.0),
                "confidence": evaluation.get("confidence", 0.0),
                "demonstrated": evaluation.get("demonstrated", []),
                "missing": evaluation.get("missing", []),
                "incorrect": evaluation.get("incorrect", []),
            }
        ]

    for verdict in verdicts:
        session.add(
            Evaluation(
                answer_id=answer_row.id,
                judge_key=str(verdict.get("judge_key", "aggregate")),
                rubric_key=evaluation.get("rubric_key"),
                score=float(verdict.get("score", 0.0)),
                communication_score=verdict.get("communication_score"),
                judge_confidence=float(verdict.get("confidence", 0.5)),
                demonstrated=list(verdict.get("demonstrated", [])),
                missing=list(verdict.get("missing", [])),
                incorrect=list(verdict.get("incorrect", [])),
                evidence_quotes=list(verdict.get("evidence_quotes", [])),
                notes=verdict.get("notes"),
            )
        )


def _persist_misconceptions(
    session: Session, record: InterviewSession, state: Mapping[str, Any]
) -> None:
    rows = state.get("misconceptions") or []
    if not rows:
        return

    existing = {
        (row.concept_key, row.belief.strip().lower()): row
        for row in session.scalars(
            select(Misconception).where(Misconception.session_id == record.id)
        )
    }

    for finding in rows:
        if not finding.get("detected"):
            continue
        identity = (
            str(finding.get("concept_key") or ""),
            str(finding.get("belief", "")).strip().lower(),
        )
        row = existing.get(identity)
        if row is not None:
            # Same belief restated under probing: count it rather than duplicating.
            row.times_observed += 1
            continue
        row = Misconception(
            candidate_id=record.candidate_id,
            session_id=record.id,
            concept_key=identity[0] or "general",
            belief=str(finding.get("belief", "")),
            correction=str(finding.get("correction", "")),
            evidence_quote=finding.get("evidence_quote"),
            severity=int(finding.get("severity", 3)),
        )
        session.add(row)
        existing[identity] = row


def _persist_study_plan(
    session: Session, record: InterviewSession, scorecard: dict[str, Any]
) -> None:
    plan_payload = scorecard.get("study_plan") or {}
    items = plan_payload.get("items") or []
    if not items:
        return

    # Only one active plan per candidate; older ones stay for history.
    for previous in session.scalars(
        select(StudyPlan).where(
            StudyPlan.candidate_id == record.candidate_id, StudyPlan.is_active.is_(True)
        )
    ):
        previous.is_active = False

    plan = StudyPlan(
        candidate_id=record.candidate_id,
        session_id=record.id,
        summary=str(plan_payload.get("summary", "")),
        is_active=True,
    )
    session.add(plan)
    session.flush()

    for item in items:
        session.add(
            StudyPlanItem(
                plan_id=plan.id,
                priority=int(item.get("priority", 1)),
                concept_key=str(item.get("concept_key", "general")),
                title=str(item.get("title", "")),
                rationale=str(item.get("rationale", "")),
                learn_items=list(item.get("learn_items", [])),
                practice_items=list(item.get("practice_items", [])),
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_result(
    record: InterviewSession,
    state: Mapping[str, Any],
    interrupt_payload: dict[str, Any] | None,
) -> TurnResult:
    question: dict[str, Any] | None = None
    clarification: dict[str, Any] | None = None

    if interrupt_payload is not None:
        if interrupt_payload.get("type") == "clarification":
            clarification = interrupt_payload
            question = _public_question(state.get("current_question"))
        else:
            question = _public_question(state.get("current_question"))

    scorecard = state.get("final_scorecard") or None
    # Coaching notes exist only in Coaching Mode; Real Mode never produces one.
    coaching = state.get("pending_coaching") or None
    return TurnResult(
        session_id=record.id,
        status=str(state.get("status", record.status)),
        question=question,
        clarification=clarification,
        coaching=coaching,
        scorecard=scorecard if scorecard else None,
        remaining_seconds=int(state.get("remaining_time", 0)),
        questions_asked=len(state.get("question_history", [])),
    )


def _public_question(question: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip interviewer-only fields.

    The candidate must never see the rubric, the concept under test, or the probe
    reason - showing them would change the answer and destroy the measurement
    (spec section 34).
    """
    if not question:
        return None
    return {
        "ordinal": question.get("ordinal"),
        "prompt_text": question.get("prompt_text"),
        "interview_type": question.get("interview_type"),
        "expects_code": question.get("expects_code", False),
        "asks_confidence": question.get("asks_confidence", False),
        "is_followup": question.get("is_followup", False),
        "asked_at": question.get("asked_at"),
    }


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
