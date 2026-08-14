"""Failure replay (spec section 24).

"Take me back to the Kafka question and let me try again." That is the single most
useful thing you can offer someone after a bad interview, and it is the reason the whole
system was built on a checkpointed state machine.

**How a fork works here, and why not the obvious way.**

The obvious approach is to rewind the LangGraph thread to a stored checkpoint id. That
works for time travel *within* one thread, but a replay is not a rewind: the original
interview must stay intact so the two attempts can be compared. So a replay is a new
session, seeded with the original's state truncated to just before the chosen question.

That has a second advantage. Truncation works off ``question_history``, ``answer_history``
and ``evidence``, all of which are ordinal-keyed and live in state, so a replay can be
started from a *completed* interview weeks later, which is exactly when someone wants it.
Rewinding a live thread could not do that.

The new graph is then positioned with ``update_state(..., as_node="ask_question")``, so
the very next step is ``wait_for_candidate``. The candidate sees the identical question,
not a regenerated variant, which is what makes the before/after comparison meaningful.
From their answer onward the interview adapts normally.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from gauntlet.db.models import InterviewQuestion, InterviewSession, ReplaySession
from gauntlet.services.interviews import TurnResult, _build_result, _sync_session, current_state
from gauntlet.services.runtime import RUNTIME

log = structlog.get_logger(__name__)


class ReplayError(Exception):
    """The interview could not be replayed from that point."""


@dataclass(frozen=True, slots=True)
class ReplayPoint:
    """A moment an interview can be restarted from."""

    ordinal: int
    prompt_text: str
    concept_keys: list[str]
    score: float | None
    is_followup: bool
    reason: str


def list_replay_points(session: Session, record: InterviewSession) -> list[ReplayPoint]:
    """Questions worth another attempt: weak answers and surfaced misconceptions."""
    scorecard = record.final_scorecard or {}
    moments = {
        int(moment["ordinal"]): moment
        for moment in scorecard.get("replay_moments", [])
        if moment.get("ordinal") is not None
    }

    points: list[ReplayPoint] = []
    for question in record.questions:
        moment = moments.get(question.ordinal)
        score = _primary_score(question)
        if moment is None and (score is None or score > 0.55):
            continue
        points.append(
            ReplayPoint(
                ordinal=question.ordinal,
                prompt_text=question.prompt_text,
                concept_keys=list(question.concept_keys),
                score=score,
                is_followup=question.is_followup,
                reason=str(moment.get("note", "Weak answer")) if moment else "Weak answer",
            )
        )
    return points


def replay_from(
    session: Session, record: InterviewSession, ordinal: int
) -> tuple[InterviewSession, TurnResult]:
    """Fork a new interview that re-asks question ``ordinal``.

    Returns the new session and the first turn, which is the original question restated.
    """
    original_state = current_state(record)
    history = list(original_state.get("question_history", []))
    if not history:
        raise ReplayError("This interview has no questions to replay.")

    target = next((item for item in history if item.get("ordinal") == ordinal), None)
    if target is None:
        raise ReplayError(f"Question {ordinal} is not part of this interview.")

    original_score = _score_for_ordinal(original_state, ordinal)

    replay_record = InterviewSession(
        candidate_id=record.candidate_id,
        resume_id=record.resume_id,
        job_description_id=record.job_description_id,
        company_id=record.company_id,
        target_role=record.target_role,
        target_level=record.target_level,
        mode=record.mode,
        interview_types=list(record.interview_types),
        planned_minutes=record.planned_minutes,
        status="awaiting_answer",
        thread_id=f"replay-{uuid.uuid4().hex}",
        plan=record.plan,
        started_at=datetime.now(UTC),
    )
    session.add(replay_record)
    session.flush()

    seeded = _truncated_state(original_state, target, ordinal, replay_record)
    config = {"configurable": {"thread_id": replay_record.thread_id}}

    # Position the new graph as though ask_question had just run, so the next step is
    # wait_for_candidate and the candidate meets the identical question.
    RUNTIME.graph.update_state(config, seeded, as_node="ask_question")
    result = RUNTIME.graph.invoke(None, config=config)

    original_question = session.scalar(
        select(InterviewQuestion).where(
            InterviewQuestion.session_id == record.id,
            InterviewQuestion.ordinal == ordinal,
        )
    )
    session.add(
        ReplaySession(
            original_session_id=record.id,
            replay_session_id=replay_record.id,
            from_question_id=original_question.id if original_question else None,
            checkpoint_id=f"ordinal:{ordinal}",
            original_score=original_score,
        )
    )

    _sync_session(session, replay_record, result)

    log.info(
        "interview.replay.started",
        original=str(record.id),
        replay=str(replay_record.id),
        ordinal=ordinal,
        original_score=original_score,
    )

    interrupts = result.get("__interrupt__")
    payload = interrupts[0].value if interrupts else None
    return replay_record, _build_result(
        replay_record, result, payload if isinstance(payload, dict) else None
    )


def settle_replay(session: Session, replay_record: InterviewSession) -> None:
    """Record the improvement once a replay finishes.

    Called when a replay session completes. The delta is the point of the whole feature:
    it is the only number in the product that says "you got better at this specific thing".
    """
    link = session.scalar(
        select(ReplaySession).where(ReplaySession.replay_session_id == replay_record.id)
    )
    if link is None or link.replay_score is not None:
        return

    state = current_state(replay_record)
    ordinal = _ordinal_from_checkpoint(link.checkpoint_id)
    replay_score = _score_for_ordinal(state, ordinal) if ordinal is not None else None
    if replay_score is None:
        return

    link.replay_score = replay_score
    if link.original_score is not None:
        link.improvement_delta = round(replay_score - float(link.original_score), 4)

    log.info(
        "interview.replay.settled",
        replay=str(replay_record.id),
        original_score=float(link.original_score) if link.original_score is not None else None,
        replay_score=replay_score,
        delta=float(link.improvement_delta) if link.improvement_delta is not None else None,
    )


def replay_history(session: Session, candidate_id: uuid.UUID) -> list[dict[str, Any]]:
    """Every replay attempt for a candidate, with the before and after."""
    rows = session.scalars(
        select(ReplaySession)
        .join(InterviewSession, ReplaySession.original_session_id == InterviewSession.id)
        .where(InterviewSession.candidate_id == candidate_id)
        .order_by(ReplaySession.created_at.desc())
    ).all()
    return [
        {
            "original_session_id": str(row.original_session_id),
            "replay_session_id": (
                str(row.replay_session_id) if row.replay_session_id else None
            ),
            "from_ordinal": _ordinal_from_checkpoint(row.checkpoint_id),
            "original_score": float(row.original_score) if row.original_score is not None else None,
            "replay_score": float(row.replay_score) if row.replay_score is not None else None,
            "improvement_delta": (
                float(row.improvement_delta) if row.improvement_delta is not None else None
            ),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _truncated_state(
    original: Mapping[str, Any],
    target: dict[str, Any],
    ordinal: int,
    replay_record: InterviewSession,
) -> dict[str, Any]:
    """Original state rewound to the moment the target question was asked.

    Everything before it is kept, because the interview genuinely happened and that
    evidence is real. Everything after it is dropped, because it is being re-run.
    """
    history = [
        item for item in original.get("question_history", []) if _ordinal(item) < ordinal
    ]
    answers = [
        item for item in original.get("answer_history", []) if _ordinal(item) < ordinal
    ]
    evidence = [item for item in original.get("evidence", []) if _ordinal(item) < ordinal]
    misconceptions = [
        item for item in original.get("misconceptions", []) if _ordinal(item) < ordinal
    ]

    restated = dict(target)
    restated["asked_at"] = datetime.now(UTC).isoformat()
    concept_keys = list(restated.get("concept_keys", []))

    return {
        **{
            key: original.get(key)
            for key in (
                "resume_text",
                "job_text",
                "resume_profile",
                "job_description",
                "target_company",
                "target_role",
                "target_level",
                "mode",
                "interview_types",
                "company_patterns",
                "interview_plan",
                "planned_minutes",
            )
        },
        "session_id": str(replay_record.id),
        "candidate_id": str(replay_record.candidate_id),
        "thread_id": replay_record.thread_id,
        # Reducer fields are replaced wholesale here rather than appended, because
        # update_state applies the same reducers as a node return would.
        "question_history": [*history, restated],
        "answer_history": answers,
        "evidence": evidence,
        "misconceptions": misconceptions,
        "interviewer_notes": [f"Replay of question {ordinal} from an earlier interview."],
        "current_question": restated,
        "pending_target": None,
        "pending_answer": None,
        "pending_clarification": None,
        "pending_coaching": None,
        "last_classification": None,
        "last_evaluation": None,
        "last_decision": None,
        "code_check": None,
        "plan_cursor": int(original.get("plan_cursor", ordinal)),
        "difficulty": int(restated.get("difficulty", 3)),
        "followups_on_concept": 0,
        "current_concept_key": concept_keys[0] if concept_keys else None,
        "started_at": datetime.now(UTC).isoformat(),
        "elapsed_time": 0,
        "remaining_time": int(replay_record.planned_minutes) * 60,
        "skill_scores": {},
        "confidence_scores": {},
        "final_scorecard": {},
        "status": "awaiting_answer",
    }


def _ordinal(item: dict[str, Any]) -> int:
    value = item.get("ordinal")
    return int(value) if value is not None else 0


def _score_for_ordinal(state: Mapping[str, Any], ordinal: int | None) -> float | None:
    if ordinal is None:
        return None
    for answer in state.get("answer_history", []):
        if _ordinal(answer) == ordinal and answer.get("score") is not None:
            return round(float(answer["score"]), 4)
    return None


def _ordinal_from_checkpoint(checkpoint_id: str | None) -> int | None:
    if not checkpoint_id or not checkpoint_id.startswith("ordinal:"):
        return None
    try:
        return int(checkpoint_id.split(":", 1)[1])
    except ValueError:
        return None


def _primary_score(question: InterviewQuestion) -> float | None:
    answer = question.answer
    if answer is None or not answer.evaluations:
        return None
    primary = next(
        (item for item in answer.evaluations if item.judge_key == "technical_accuracy"),
        answer.evaluations[0],
    )
    return round(float(primary.score), 3)
