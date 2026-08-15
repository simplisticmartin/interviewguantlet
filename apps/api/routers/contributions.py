"""Contributing questions, and moderating what gets contributed (spec sections 37, 38).

Two audiences on one resource. A candidate can offer a question and see what happened to
their own submissions. A moderator can see the queue and rule on it. The split is enforced
by the dependency on each route, not by a field check inside the handler, so adding a
route cannot accidentally omit the gate.

Nothing here publishes on its own. ``POST /contributions`` always ends in a pending row,
and the response says so in words, because a contributor who assumes their question went
live will report the feature as broken when they cannot find it.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from apps.api.deps import RateLimiter, get_current_candidate, get_db
from apps.api.schemas import (
    ContributionRequest,
    ContributionResponse,
    ModerationDecision,
    SubmissionView,
)
from gauntlet.db.models import Candidate, QuestionSubmission, User
from gauntlet.ingestion.pipeline import Submission
from gauntlet.services import contributions
from gauntlet.services.contributions import ContributionError

router = APIRouter(tags=["contributions"])

# Contribution is cheap to abuse and expensive to moderate, so it is throttled harder
# than ordinary reads: every accepted submission costs a human's attention.
contribution_rate_limit = RateLimiter(limit=10, window_seconds=300.0)


def get_current_moderator(
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> User:
    """Authorise a reviewer.

    Returns 404 rather than 403 for non-moderators: the existence of a moderation queue
    is not something an ordinary account needs confirmed.
    """
    user = session.get(User, candidate.user_id)
    if user is None or not user.is_moderator:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    return user


def _to_view(row: QuestionSubmission) -> SubmissionView:
    return SubmissionView(
        id=row.id,
        question=row.question,
        company_slug=row.company_slug,
        level=row.level,
        interview_type=row.interview_type,
        concept_keys=list(row.concept_keys),
        difficulty=row.difficulty,
        status=row.status,
        safety_verdict=row.safety_verdict,
        safety_findings=list(row.safety_findings),
        review_reasons=list(row.review_reasons),
        near_duplicates=list(row.near_duplicates),
        duplicate_of_slug=row.duplicate_of_slug,
        created_at=row.created_at,
        published_question_id=row.published_question_id,
    )


@router.post(
    "/contributions",
    response_model=ContributionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(contribution_rate_limit)],
)
def contribute(
    payload: ContributionRequest,
    session: Session = Depends(get_db),
    candidate: Candidate = Depends(get_current_candidate),
) -> ContributionResponse:
    """Offer a question for review. 202, not 201: nothing is created in the corpus."""
    submission = Submission(
        question=payload.question,
        company=payload.company,
        role=payload.role,
        level=payload.level,
        interview_round=payload.interview_round,
        asked_on=payload.asked_on,
        notes=payload.notes,
        difficulty=payload.difficulty,
        contributor_id=str(candidate.id),
    )

    try:
        row = contributions.submit(session, candidate, submission)
    except ContributionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"detail": str(exc), "reasons": exc.reasons},
        ) from exc

    if row.status == "duplicate":
        message = (
            "Thanks. This one is already in the bank, so it has been recorded as another "
            "sighting rather than a new question."
        )
    else:
        message = (
            "Thanks. Your question is queued for review and is not published yet. "
            "Personal details are removed before a reviewer sees it."
        )

    return ContributionResponse(
        id=row.id,
        status=row.status,
        question=row.question,
        concept_keys=list(row.concept_keys),
        interview_type=row.interview_type,
        difficulty=row.difficulty,
        safety_verdict=row.safety_verdict,
        review_reasons=list(row.review_reasons),
        duplicate_of=row.duplicate_of_slug,
        message=message,
    )


@router.get("/contributions/mine", response_model=list[SubmissionView])
def my_contributions(
    session: Session = Depends(get_db),
    candidate: Candidate = Depends(get_current_candidate),
) -> list[SubmissionView]:
    return [_to_view(row) for row in contributions.my_submissions(session, candidate)]


@router.get("/moderation/submissions", response_model=list[SubmissionView])
def review_queue(
    submission_status: Annotated[
        str, Query(pattern="^(pending|approved|rejected|duplicate)$")
    ] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    session: Session = Depends(get_db),
    _: User = Depends(get_current_moderator),
) -> list[SubmissionView]:
    rows = contributions.queue(session, status=submission_status, limit=limit)
    return [_to_view(row) for row in rows]


@router.post("/moderation/submissions/{submission_id}", response_model=SubmissionView)
def decide(
    submission_id: uuid.UUID,
    decision: ModerationDecision,
    session: Session = Depends(get_db),
    moderator: User = Depends(get_current_moderator),
) -> SubmissionView:
    row = session.get(QuestionSubmission, submission_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")

    try:
        if decision.decision == "approve":
            contributions.approve(
                session,
                row,
                moderator,
                note=decision.note,
                interview_type=decision.interview_type,
                concept_keys=decision.concept_keys,
                difficulty=decision.difficulty,
            )
        else:
            contributions.reject(session, row, moderator, note=decision.note)
    except ContributionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    return _to_view(row)
