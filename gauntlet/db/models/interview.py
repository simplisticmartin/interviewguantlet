"""Interview sessions, the questions actually asked, answers, and evaluations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gauntlet.db.base import Base, TimestampMixin


class InterviewSession(Base, TimestampMixin):
    __tablename__ = "interview_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("resumes.id", ondelete="SET NULL")
    )
    job_description_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("job_descriptions.id", ondelete="SET NULL")
    )
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("companies.id", ondelete="SET NULL")
    )

    target_role: Mapped[str] = mapped_column(String(200), nullable=False)
    target_level: Mapped[str] = mapped_column(String(60), nullable=False, default="senior")
    mode: Mapped[str] = mapped_column(String(40), nullable=False, default="real")
    interview_types: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    planned_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=20)

    status: Mapped[str] = mapped_column(String(30), nullable=False, default="created", index=True)
    # LangGraph checkpointer thread. This is what makes recovery and replay possible.
    thread_id: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)

    plan: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    final_scorecard: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    questions: Mapped[list[InterviewQuestion]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="InterviewQuestion.ordinal",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('created', 'in_progress', 'awaiting_answer', 'completed', 'abandoned')",
            name="status_enum",
        ),
        CheckConstraint(
            "mode IN ('real', 'coaching', 'rapid_fire', 'coding', 'system_design', "
            "'resume_defense', 'behavioral', 'full_loop')",
            name="mode_enum",
        ),
    )


class InterviewQuestion(Base, TimestampMixin):
    """One question put to the candidate, including adaptive follow-ups."""

    __tablename__ = "interview_questions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interview_questions.id", ondelete="CASCADE")
    )
    question_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("questions.id", ondelete="SET NULL")
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    interview_type: Mapped[str] = mapped_column(String(60), nullable=False)
    agent_key: Mapped[str] = mapped_column(String(60), nullable=False)
    concept_keys: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    rubric_key: Mapped[str | None] = mapped_column(String(200))
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    is_followup: Mapped[bool] = mapped_column(default=False, nullable=False)
    probe_reason: Mapped[str | None] = mapped_column(Text)
    asks_confidence: Mapped[bool] = mapped_column(default=False, nullable=False)

    # Checkpoint immediately BEFORE this question was asked - the anchor for
    # "replay from here" (spec section 24).
    checkpoint_id: Mapped[str | None] = mapped_column(String(120))
    asked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[InterviewSession] = relationship(back_populates="questions")
    answer: Mapped[CandidateAnswer | None] = relationship(
        back_populates="question", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("session_id", "ordinal", name="uq_session_ordinal"),
        Index("ix_interview_questions_session_ordinal", "session_id", "ordinal"),
    )


class CandidateAnswer(Base, TimestampMixin):
    __tablename__ = "candidate_answers"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    interview_question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_questions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    code: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(40))
    # Candidate's self-rated confidence, 1-5 (spec section 22). Null when not asked.
    self_confidence: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    hints_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    question: Mapped[InterviewQuestion] = relationship(back_populates="answer")
    evaluations: Mapped[list[Evaluation]] = relationship(
        back_populates="answer", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "self_confidence IS NULL OR self_confidence BETWEEN 1 AND 5",
            name="self_confidence_range",
        ),
    )


class Evaluation(Base, TimestampMixin):
    """One judge's verdict on one answer.

    Several rows per answer is the normal case: spec section 19 wants independent
    judges whose disagreement is itself a measurable signal.
    """

    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    answer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidate_answers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    judge_key: Mapped[str] = mapped_column(String(60), nullable=False)
    rubric_key: Mapped[str | None] = mapped_column(String(200))

    score: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    communication_score: Mapped[float | None] = mapped_column(Numeric(4, 3))
    judge_confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.5)

    demonstrated: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    missing: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    incorrect: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    evidence_quotes: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    notes: Mapped[str | None] = mapped_column(Text)

    # Reproducibility (spec section 42): which prompt and model produced this verdict.
    prompt_name: Mapped[str | None] = mapped_column(String(120))
    prompt_version: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(120))

    answer: Mapped[CandidateAnswer] = relationship(back_populates="evaluations")

    __table_args__ = (
        UniqueConstraint("answer_id", "judge_key", name="uq_answer_judge"),
        CheckConstraint("score BETWEEN 0 AND 1", name="score_range"),
    )


class ReplaySession(Base, TimestampMixin):
    """Links a replay attempt back to the moment it forked from (spec section 24)."""

    __tablename__ = "replay_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    original_session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    replay_session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="SET NULL")
    )
    from_question_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interview_questions.id", ondelete="SET NULL")
    )
    checkpoint_id: Mapped[str] = mapped_column(String(120), nullable=False)
    original_score: Mapped[float | None] = mapped_column(Numeric(4, 3))
    replay_score: Mapped[float | None] = mapped_column(Numeric(4, 3))
    improvement_delta: Mapped[float | None] = mapped_column(Numeric(5, 3))
