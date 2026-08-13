"""Interview lifecycle endpoints."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from apps.api.deps import get_current_candidate, get_db, interview_rate_limit
from apps.api.schemas import (
    AnswerRequest,
    CreateInterviewRequest,
    InterviewDetail,
    InterviewSummary,
    QuestionView,
    TranscriptEntry,
    TurnResponse,
)
from gauntlet.db.models import Candidate, Company, InterviewSession
from gauntlet.schemas import AnswerPayload, InterviewType
from gauntlet.services.interviews import (
    InterviewError,
    StartRequest,
    TurnResult,
    checkpoint_history,
    finish_interview,
    start_interview,
    submit_answer,
)
from gauntlet.services.runtime import RUNTIME

log = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/interviews", tags=["interviews"], dependencies=[Depends(interview_rate_limit)]
)

DEFAULT_TYPES = [
    InterviewType.JAVA,
    InterviewType.SPRING,
    InterviewType.DATABASE,
    InterviewType.DISTRIBUTED,
    InterviewType.SYSTEM_DESIGN,
]


@router.post("", response_model=TurnResponse, status_code=status.HTTP_201_CREATED)
def create_interview(
    payload: CreateInterviewRequest,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> TurnResponse:
    try:
        result = start_interview(
            session,
            candidate.id,
            StartRequest(
                resume_id=payload.resume_id,
                job_description_id=payload.job_description_id,
                target_role=payload.target_role,
                target_level=payload.target_level,
                mode=payload.mode,
                interview_types=payload.interview_types or DEFAULT_TYPES,
                minutes=payload.minutes,
                company_slug=payload.company,
            ),
        )
    except InterviewError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _to_response(result)


@router.post("/{session_id}/answer", response_model=TurnResponse)
def answer(
    session_id: uuid.UUID,
    payload: AnswerRequest,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> TurnResponse:
    record = _owned_session(session, candidate, session_id)
    if not payload.text.strip() and not payload.code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="An answer cannot be empty."
        )
    try:
        result = submit_answer(session, record, AnswerPayload.model_validate(payload.model_dump()))
    except InterviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _to_response(result)


@router.post("/{session_id}/answer/stream")
async def answer_stream(
    session_id: uuid.UUID,
    payload: AnswerRequest,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> EventSourceResponse:
    """Stream pipeline progress while the answer is processed (spec section 32).

    A turn runs several agents - classify, four judges, misconception check, router,
    author the next question - and takes long enough that silence feels broken. This
    reports which stage is running without leaking any evaluation result: node names
    only, never scores.
    """
    record = _owned_session(session, candidate, session_id)
    if not payload.text.strip() and not payload.code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="An answer cannot be empty."
        )

    from langgraph.types import Command

    from gauntlet.services.interviews import _sync_session

    answer_payload = AnswerPayload.model_validate(payload.model_dump())
    config = {"configurable": {"thread_id": record.thread_id}}

    # Candidate-facing stage labels. Deliberately vague about evaluation: telling
    # someone "grading your answer: 0.3" mid-interview would change how they behave.
    stage_labels = {
        "classify_response": "Listening",
        "check_code": "Reading your code",
        "evaluate_answer": "Considering your answer",
        "update_skill_graph": "Considering your answer",
        "misconception_check": "Considering your answer",
        "adaptive_router": "Deciding what to ask next",
        "select_question": "Deciding what to ask next",
        "ask_question": "Preparing the next question",
        "answer_clarification": "Answering your question",
        "report": "Writing your scorecard",
    }

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        final_state: dict = {}
        try:
            for chunk in RUNTIME.graph.stream(
                Command(resume=answer_payload.model_dump(mode="json")),
                config=config,
                stream_mode="updates",
            ):
                for node_name, update in chunk.items():
                    if node_name == "__interrupt__":
                        continue
                    if isinstance(update, dict):
                        final_state.update(update)
                    label = stage_labels.get(node_name)
                    if label:
                        yield {
                            "event": "stage",
                            "data": json.dumps({"node": node_name, "label": label}),
                        }

            snapshot = RUNTIME.graph.get_state(config)
            state = dict(snapshot.values) if snapshot else final_state
            _sync_session(session, record, state)
            session.commit()

            interrupts = getattr(snapshot, "interrupts", ()) if snapshot else ()
            payload_value = interrupts[0].value if interrupts else None
            result = _build_from_state(record, state, payload_value)
            yield {"event": "turn", "data": _to_response(result).model_dump_json()}
        except Exception as exc:  # surface failures to the client rather than hanging
            session.rollback()
            log.exception("interview.stream.failed", session=str(record.id))
            yield {"event": "error", "data": json.dumps({"detail": str(exc)[:300]})}

    return EventSourceResponse(event_stream())


@router.post("/{session_id}/finish", response_model=TurnResponse)
def finish(
    session_id: uuid.UUID,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> TurnResponse:
    record = _owned_session(session, candidate, session_id)
    result = finish_interview(session, record)
    return _to_response(result)


@router.get("", response_model=list[InterviewSummary])
def list_interviews(
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> list[InterviewSummary]:
    rows = session.scalars(
        select(InterviewSession)
        .where(InterviewSession.candidate_id == candidate.id)
        .order_by(InterviewSession.created_at.desc())
    ).all()
    return [_summary(session, row) for row in rows]


@router.get("/{session_id}", response_model=InterviewDetail)
def get_interview(
    session_id: uuid.UUID,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> InterviewDetail:
    record = _owned_session(session, candidate, session_id)
    summary = _summary(session, record)
    is_complete = record.status == "completed"

    transcript = [
        TranscriptEntry(
            ordinal=question.ordinal,
            prompt_text=question.prompt_text,
            interview_type=question.interview_type,
            is_followup=question.is_followup,
            answer_text=question.answer.text if question.answer else None,
            self_confidence=question.answer.self_confidence if question.answer else None,
            # Per-question scores stay hidden until the interview ends: showing a live
            # score changes how candidates answer the next one (spec section 34).
            score=_question_score(question) if is_complete else None,
            concept_keys=list(question.concept_keys) if is_complete else [],
        )
        for question in record.questions
    ]

    return InterviewDetail(
        **summary.model_dump(),
        transcript=transcript,
        scorecard=record.final_scorecard or None,
        plan=record.plan if is_complete else {},
    )


@router.get("/{session_id}/checkpoints")
def list_checkpoints(
    session_id: uuid.UUID,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    """Checkpoints this interview can be rewound to.

    The data failure replay needs is captured from day one; the replay endpoint that
    forks a new interview from a checkpoint lands in roadmap phase 6.
    """
    record = _owned_session(session, candidate, session_id)
    return {
        "session_id": str(record.id),
        "durable": RUNTIME.durable_checkpoints,
        "checkpoints": checkpoint_history(record),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _owned_session(
    session: Session, candidate: Candidate, session_id: uuid.UUID
) -> InterviewSession:
    record = session.get(InterviewSession, session_id)
    if record is None or record.candidate_id != candidate.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found.")
    return record


def _question_score(question: object) -> float | None:
    answer = getattr(question, "answer", None)
    if answer is None or not answer.evaluations:
        return None
    primary = next(
        (item for item in answer.evaluations if item.judge_key == "technical_accuracy"),
        answer.evaluations[0],
    )
    return round(float(primary.score), 3)


def _summary(session: Session, record: InterviewSession) -> InterviewSummary:
    scorecard = record.final_scorecard or {}
    company_name: str | None = None
    if record.company_id:
        company = session.get(Company, record.company_id)
        company_name = company.name if company else None

    return InterviewSummary(
        id=record.id,
        target_role=record.target_role,
        target_level=record.target_level,
        mode=record.mode,
        status=record.status,
        company=company_name,
        planned_minutes=record.planned_minutes,
        questions_asked=len(record.questions),
        overall=scorecard.get("overall"),
        recommendation=(scorecard.get("committee") or {}).get("recommendation"),
        started_at=record.started_at,
        ended_at=record.ended_at,
    )


def _build_from_state(
    record: InterviewSession, state: dict, interrupt_value: object
) -> TurnResult:
    from gauntlet.services.interviews import _build_result

    payload = interrupt_value if isinstance(interrupt_value, dict) else None
    return _build_result(record, state, payload)


def _to_response(result: TurnResult) -> TurnResponse:
    return TurnResponse(
        session_id=result.session_id,
        status=result.status,
        question=QuestionView.model_validate(result.question) if result.question else None,
        clarification=result.clarification,
        scorecard=result.scorecard,
        remaining_seconds=result.remaining_seconds,
        questions_asked=result.questions_asked,
    )
