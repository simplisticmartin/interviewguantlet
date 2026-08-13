"""Persistent candidate knowledge: skill graph, misconceptions, study plans."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gauntlet.db.models.identity import Candidate


class CandidateSkillState(Base, TimestampMixin):
    """Current belief about one concept for one candidate (spec section 21).

    ``mastery`` and ``confidence`` are separate on purpose: the interesting quadrant
    is low mastery + high confidence, which is a misconception, not a gap.
    """

    __tablename__ = "candidate_skill_states"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)

    mastery: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.0)
    self_confidence: Mapped[float | None] = mapped_column(Numeric(4, 3))
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_evidence_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Spaced repetition scheduling (spec section 30).
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    repetition_interval_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    candidate: Mapped[Candidate] = relationship(back_populates="skill_states")

    __table_args__ = (
        UniqueConstraint("candidate_id", "concept_key", name="uq_candidate_concept"),
        CheckConstraint("mastery BETWEEN 0 AND 1", name="mastery_range"),
    )


class SkillEvidence(Base, TimestampMixin):
    """Append-only observations. Skill state is derivable from these rows.

    Keeping raw evidence rather than only the aggregate is what lets the mastery
    formula be swapped for IRT / Bayesian knowledge tracing later (spec section 23)
    and recomputed over history.
    """

    __tablename__ = "skill_evidence"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="SET NULL")
    )
    answer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("candidate_answers.id", ondelete="SET NULL")
    )

    score: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    self_confidence: Mapped[int | None] = mapped_column(Integer)
    hints_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_followup: Mapped[bool] = mapped_column(default=False, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_skill_evidence_candidate_concept", "candidate_id", "concept_key"),)


class Misconception(Base, TimestampMixin):
    """A confidently-wrong belief (spec section 22, the priority quadrant)."""

    __tablename__ = "misconceptions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="SET NULL")
    )
    concept_key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    belief: Mapped[str] = mapped_column(Text, nullable=False)
    correction: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_quote: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")
    times_observed: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class StudyPlan(Base, TimestampMixin):
    __tablename__ = "study_plans"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="SET NULL")
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    items: Mapped[list[StudyPlanItem]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="StudyPlanItem.priority"
    )


class StudyPlanItem(Base, TimestampMixin):
    __tablename__ = "study_plan_items"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("study_plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    concept_key: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    learn_items: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    practice_items: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    reattempt_question_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("interview_questions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="todo")

    plan: Mapped[StudyPlan] = relationship(back_populates="items")
