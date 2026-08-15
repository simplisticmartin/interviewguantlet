"""Persistence and moderation for user-contributed questions (spec sections 37, 38).

The pipeline in ``gauntlet.ingestion.pipeline`` decides *what* a submission is; this
module is where that decision becomes a row, and where a moderator's judgement turns a
pending row into a corpus question.

Two rules are enforced here rather than left to callers:

1. **Only the redacted text is stored.** The raw submission is never persisted. Keeping
   it "just in case" would defeat the screening step entirely, since the personal data
   would still be sitting in the database.
2. **Approval is the only path into the corpus.** Nothing writes to ``questions`` except
   :func:`approve`, and what it writes always carries ``question_origin="user_submitted"``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from gauntlet.db.models import Candidate, Question, QuestionSubmission, User
from gauntlet.ingestion.pipeline import Outcome, PipelineResult, Submission, process

log = structlog.get_logger(__name__)


class ContributionError(RuntimeError):
    """A submission cannot be accepted, with a reason fit to show the contributor."""

    def __init__(self, message: str, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []


def submit(
    session: Session, candidate: Candidate, submission: Submission
) -> QuestionSubmission:
    """Screen a submission and record it for review.

    Raises :class:`ContributionError` when the pipeline refuses it, so the caller can
    return the reasons rather than storing something that was never acceptable.
    """
    result = process(submission)

    if result.outcome is Outcome.REJECTED:
        log.info("contribution.rejected", candidate_id=str(candidate.id))
        raise ContributionError("Submission was not accepted.", result.reasons)

    row = _to_row(candidate, submission, result)
    session.add(row)
    session.flush()
    log.info(
        "contribution.recorded",
        submission_id=str(row.id),
        status=row.status,
        verdict=row.safety_verdict,
    )
    return row


def _to_row(
    candidate: Candidate, submission: Submission, result: PipelineResult
) -> QuestionSubmission:
    return QuestionSubmission(
        candidate_id=candidate.id,
        # Redacted text only. The original never reaches the database.
        question=result.question,
        notes=None,
        company_slug=submission.company,
        role_family=submission.role,
        level=submission.level,
        interview_round=submission.interview_round,
        interview_type=result.interview_type.value if result.interview_type else None,
        concept_keys=list(result.concept_keys),
        difficulty=result.difficulty,
        asked_on=submission.asked_on,
        status="duplicate" if result.outcome is Outcome.DUPLICATE else "pending",
        safety_verdict=result.safety.verdict.value,
        safety_findings=[finding.kind.value for finding in result.safety.findings],
        review_reasons=list(result.reasons),
        near_duplicates=[list(pair) for pair in result.near_duplicates],
        duplicate_of_slug=result.duplicate_of,
        provenance=result.provenance,
    )


def queue(
    session: Session, *, status: str = "pending", limit: int = 50
) -> list[QuestionSubmission]:
    """The moderation queue, oldest first so nothing waits indefinitely."""
    stmt: Select[tuple[QuestionSubmission]] = (
        select(QuestionSubmission)
        .where(QuestionSubmission.status == status)
        .order_by(QuestionSubmission.created_at.asc())
        .limit(limit)
    )
    return list(session.execute(stmt).scalars())


def my_submissions(
    session: Session, candidate: Candidate, *, limit: int = 50
) -> list[QuestionSubmission]:
    stmt: Select[tuple[QuestionSubmission]] = (
        select(QuestionSubmission)
        .where(QuestionSubmission.candidate_id == candidate.id)
        .order_by(QuestionSubmission.created_at.desc())
        .limit(limit)
    )
    return list(session.execute(stmt).scalars())


def _settle(
    submission: QuestionSubmission, reviewer: User, status: str, note: str | None
) -> None:
    submission.status = status
    submission.reviewed_by = reviewer.id
    submission.reviewed_at = datetime.now(UTC)
    submission.reviewer_note = note


def reject(
    session: Session, submission: QuestionSubmission, reviewer: User, note: str | None = None
) -> QuestionSubmission:
    _settle(submission, reviewer, "rejected", note)
    session.flush()
    log.info("contribution.rejected_by_review", submission_id=str(submission.id))
    return submission


def approve(
    session: Session,
    submission: QuestionSubmission,
    reviewer: User,
    *,
    note: str | None = None,
    interview_type: str | None = None,
    concept_keys: list[str] | None = None,
    difficulty: int | None = None,
) -> Question:
    """Promote a reviewed submission into the corpus.

    The reviewer can correct the automatic tagging on the way through, which is the point
    of having a human in the loop: the tagger is conservative and often leaves a question
    on a broad concept that a person can place properly.
    """
    if submission.status != "pending":
        raise ContributionError(f"Submission is already {submission.status}.")

    resolved_type = interview_type or submission.interview_type
    if not resolved_type:
        raise ContributionError(
            "Cannot publish without an interview type; set one while approving."
        )
    resolved_concepts = concept_keys or submission.concept_keys
    if not resolved_concepts:
        raise ContributionError(
            "Cannot publish without at least one concept; tag it while approving."
        )

    question = Question(
        id=uuid.uuid4(),
        question=submission.question,
        follow_ups=[],
        level=submission.level,
        interview_type=resolved_type,
        concept_keys=list(resolved_concepts),
        topics=[],
        difficulty=difficulty or submission.difficulty,
        # Provenance is not negotiable: a contributed question stays distinguishable from
        # the authored corpus forever, and confidence starts low because one person
        # saying they were asked something is weak evidence.
        question_origin="user_submitted",
        source_type="user_contribution",
        copyright_status="contributor_asserted_original",
        source_date=submission.asked_on,
        based_on_patterns=[],
        confidence=0.35,
        is_active=True,
    )
    session.add(question)
    session.flush()

    _settle(submission, reviewer, "approved", note)
    submission.published_question_id = question.id
    session.flush()

    log.info(
        "contribution.approved",
        submission_id=str(submission.id),
        question_id=str(question.id),
        reviewer=str(reviewer.id),
    )
    return question
