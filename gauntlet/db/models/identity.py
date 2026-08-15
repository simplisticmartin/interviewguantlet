"""Users, candidates, resumes, and the claims extracted from resumes."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gauntlet.db.base import Base, TimestampMixin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gauntlet.db.models.interview import InterviewSession
    from gauntlet.db.models.learning import CandidateSkillState


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    # Moderators review user-contributed questions (spec section 38). A flag on the user
    # rather than an allowlist in config, so granting review rights is an audited row
    # change instead of a redeploy.
    is_moderator: Mapped[bool] = mapped_column(default=False, nullable=False)

    candidate: Mapped[Candidate | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Candidate(Base, TimestampMixin):
    """The interviewing persona attached to a user account.

    Kept separate from ``users`` so a future org/recruiter account can own several
    candidate profiles without reshaping auth.
    """

    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    target_company_slug: Mapped[str | None] = mapped_column(String(120))
    target_role: Mapped[str | None] = mapped_column(String(160))
    target_level: Mapped[str | None] = mapped_column(String(60))

    user: Mapped[User] = relationship(back_populates="candidate")
    resumes: Mapped[list[Resume]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    sessions: Mapped[list[InterviewSession]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    skill_states: Mapped[list[CandidateSkillState]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )


class Resume(Base, TimestampMixin):
    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(400), nullable=False)
    content_type: Mapped[str] = mapped_column(String(160), nullable=False, default="text/plain")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Structured output of the parser agent: skills, roles, projects, years of experience.
    profile: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    is_primary: Mapped[bool] = mapped_column(default=True, nullable=False)

    candidate: Mapped[Candidate] = relationship(back_populates="resumes")
    claims: Mapped[list[ResumeClaim]] = relationship(
        back_populates="resume", cascade="all, delete-orphan"
    )


class ResumeClaim(Base, TimestampMixin):
    """A single verifiable assertion lifted out of a resume (spec section 14).

    ``support_level`` is *not* a judgement about honesty - it records how much
    interview evidence backs the claim, which is what the report is allowed to say.
    """

    __tablename__ = "resume_claims"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    resume_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(60), nullable=False, default="experience")
    technologies: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    concept_keys: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    has_metric: Mapped[bool] = mapped_column(default=False, nullable=False)
    probe_priority: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    support_level: Mapped[str | None] = mapped_column(String(40))
    support_score: Mapped[float | None] = mapped_column(Numeric(4, 3))

    resume: Mapped[Resume] = relationship(back_populates="claims")


Index("ix_resume_claims_priority", ResumeClaim.resume_id, ResumeClaim.probe_priority)
