"""Company / role / concept catalog and the interview-question corpus.

The question corpus carries provenance on every row (spec section 7) so the app can
always answer "why do you believe this?" - and can distinguish a question actually
reported by a candidate from one Gauntlet generated in the style of a company.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Date,
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

# Fixed at the schema level: changing it is a migration, not a config flip.
EMBEDDING_DIM = 1536


class Company(Base, TimestampMixin):
    """Normalised company entity - companies are data, never hardcoded (spec section 6)."""

    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    sector: Mapped[str] = mapped_column(String(80), nullable=False, default="technology")
    aliases: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    # Estimated interview-type distribution by level, e.g. {"senior": {"dsa": 0.3, ...}}
    interview_mix: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    notes: Mapped[str | None] = mapped_column(Text)

    occurrences: Mapped[list[CompanyQuestionOccurrence]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )


class Role(Base, TimestampMixin):
    __tablename__ = "roles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    role_family: Mapped[str] = mapped_column(String(120), nullable=False)
    levels: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    core_concepts: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)


class JobDescription(Base, TimestampMixin):
    __tablename__ = "job_descriptions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("companies.id", ondelete="SET NULL")
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(String(240))
    level: Mapped[str | None] = mapped_column(String(60))
    # Structured analysis: required skills, weighted concept keys, responsibilities.
    analysis: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class Concept(Base, TimestampMixin):
    """A node in the hierarchical skill taxonomy (spec section 21).

    ``key`` is a dotted path, e.g. ``java.concurrency.concurrent_hashmap``. The parent
    relationship is derived from the key so the tree can never disagree with itself.
    """

    __tablename__ = "concepts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    parent_key: Mapped[str | None] = mapped_column(String(200), index=True)
    display_name: Mapped[str] = mapped_column(String(240), nullable=False)
    domain: Mapped[str] = mapped_column(String(80), nullable=False)
    interview_type: Mapped[str] = mapped_column(String(60), nullable=False, default="technical")
    difficulty_floor: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    difficulty_ceiling: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    aliases: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)


class QuestionFamily(Base, TimestampMixin):
    """Canonical grouping for semantically equivalent questions (spec section 8).

    "Two Sum", "return two indices summing to target", and "find a pair adding to K"
    collapse into one family so question counts cannot be inflated by rewording.
    """

    __tablename__ = "question_families"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    canonical_text: Mapped[str] = mapped_column(Text, nullable=False)
    topics: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    variant_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    questions: Mapped[list[Question]] = relationship(back_populates="family")


class Question(Base, TimestampMixin):
    __tablename__ = "questions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    family_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("question_families.id", ondelete="SET NULL"), index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    follow_ups: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    role_family: Mapped[str] = mapped_column(
        String(120), nullable=False, default="Software Engineering"
    )
    level: Mapped[str | None] = mapped_column(String(60), index=True)
    interview_type: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    concept_keys: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    topics: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    rubric_key: Mapped[str | None] = mapped_column(String(200), index=True)

    # --- Provenance (spec section 7). Never claim a company asked a generated question. ---
    question_origin: Mapped[str] = mapped_column(String(40), nullable=False, default="generated")
    source_type: Mapped[str] = mapped_column(
        String(60), nullable=False, default="gauntlet_authored"
    )
    source_url: Mapped[str | None] = mapped_column(Text)
    source_date: Mapped[date | None] = mapped_column(Date)
    based_on_patterns: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.5)
    copyright_status: Mapped[str] = mapped_column(String(60), nullable=False, default="original")

    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    family: Mapped[QuestionFamily | None] = relationship(back_populates="questions")
    occurrences: Mapped[list[CompanyQuestionOccurrence]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty_range"),
        CheckConstraint(
            "question_origin IN ('observed', 'generated', 'user_submitted')",
            name="origin_enum",
        ),
        Index("ix_questions_type_difficulty", "interview_type", "difficulty"),
    )


class CompanyQuestionOccurrence(Base, TimestampMixin):
    """Evidence that a question (or its family) showed up at a company.

    Deliberately *not* a boolean: spec section 9 requires representing uncertainty and
    section 10 requires time decay, so we store counts and dates, not claims.
    """

    __tablename__ = "company_question_occurrences"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    level: Mapped[str | None] = mapped_column(String(60))
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_reported_on: Mapped[date | None] = mapped_column(Date)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=0.5)

    company: Mapped[Company] = relationship(back_populates="occurrences")
    question: Mapped[Question] = relationship(back_populates="occurrences")

    __table_args__ = (
        UniqueConstraint("company_id", "question_id", "level", name="uq_company_question_level"),
    )


class QuestionSubmission(Base, TimestampMixin):
    """A user-contributed question awaiting human review (spec sections 37 and 38).

    Kept in its own table rather than as a draft row in ``questions``. A pending
    submission is not a question yet: it has been screened but not approved, and putting
    it in the corpus table would make "is this reviewed?" a filter that some future query
    forgets to apply. Separate tables make the unreviewed state impossible to serve by
    accident.

    The screening result is stored alongside the text, so a reviewer sees what the filter
    found and why it escalated, rather than being asked to re-derive it.
    """

    __tablename__ = "question_submissions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("candidates.id", ondelete="SET NULL"), index=True
    )

    # Post-redaction text. The raw submission is deliberately never persisted: storing it
    # would keep the personal data the screening step exists to remove.
    question: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    company_slug: Mapped[str | None] = mapped_column(String(120), index=True)
    role_family: Mapped[str | None] = mapped_column(String(120))
    level: Mapped[str | None] = mapped_column(String(60))
    interview_round: Mapped[str | None] = mapped_column(String(60))
    interview_type: Mapped[str | None] = mapped_column(String(60))
    concept_keys: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    difficulty: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    asked_on: Mapped[date | None] = mapped_column(Date)

    # --- Review state ------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="pending", index=True
    )
    safety_verdict: Mapped[str] = mapped_column(String(20), nullable=False, default="accept")
    safety_findings: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    review_reasons: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    near_duplicates: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    duplicate_of_slug: Mapped[str | None] = mapped_column(String(200))

    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewer_note: Mapped[str | None] = mapped_column(Text)

    # Set once a reviewer promotes the submission into the corpus.
    published_question_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("questions.id", ondelete="SET NULL")
    )

    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'duplicate')",
            name="submission_status_enum",
        ),
        CheckConstraint(
            "safety_verdict IN ('accept', 'review', 'block')", name="submission_verdict_enum"
        ),
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="submission_difficulty_range"),
        Index("ix_question_submissions_status_created", "status", "created_at"),
    )
