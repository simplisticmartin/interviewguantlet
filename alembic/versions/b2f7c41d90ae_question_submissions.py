"""question submissions review queue

Adds the queue that user-contributed questions land in (spec sections 37 and 38).
Nothing here writes to ``questions``: a submission becomes a corpus question only when a
reviewer promotes it, which is a separate write that sets ``published_question_id``.

Revision ID: b2f7c41d90ae
Revises: 81ae954d38d7
Create Date: 2026-08-14 22:41:10.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b2f7c41d90ae"
down_revision: str | None = "81ae954d38d7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing users default to non-moderator: review rights are granted deliberately.
    op.add_column(
        "users",
        sa.Column(
            "is_moderator", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )
    op.create_table(
        "question_submissions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("company_slug", sa.String(length=120), nullable=True),
        sa.Column("role_family", sa.String(length=120), nullable=True),
        sa.Column("level", sa.String(length=60), nullable=True),
        sa.Column("interview_round", sa.String(length=60), nullable=True),
        sa.Column("interview_type", sa.String(length=60), nullable=True),
        sa.Column("concept_keys", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("difficulty", sa.Integer(), nullable=False),
        sa.Column("asked_on", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("safety_verdict", sa.String(length=20), nullable=False),
        sa.Column("safety_findings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("review_reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("near_duplicates", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("duplicate_of_slug", sa.String(length=200), nullable=True),
        sa.Column("reviewed_by", sa.Uuid(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("published_question_id", sa.Uuid(), nullable=True),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "difficulty BETWEEN 1 AND 5",
            name=op.f("ck_question_submissions_submission_difficulty_range"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'duplicate')",
            name=op.f("ck_question_submissions_submission_status_enum"),
        ),
        sa.CheckConstraint(
            "safety_verdict IN ('accept', 'review', 'block')",
            name=op.f("ck_question_submissions_submission_verdict_enum"),
        ),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["candidates.id"],
            name=op.f("fk_question_submissions_candidate_id_candidates"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["published_question_id"],
            ["questions.id"],
            name=op.f("fk_question_submissions_published_question_id_questions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["reviewed_by"],
            ["users.id"],
            name=op.f("fk_question_submissions_reviewed_by_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_question_submissions")),
    )
    op.create_index(
        op.f("ix_question_submissions_candidate_id"),
        "question_submissions",
        ["candidate_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_question_submissions_company_slug"),
        "question_submissions",
        ["company_slug"],
        unique=False,
    )
    op.create_index(
        op.f("ix_question_submissions_status"), "question_submissions", ["status"], unique=False
    )
    # The moderation queue is always read as "oldest pending first", so the index matches
    # that access pattern rather than leaving it to a filter plus sort.
    op.create_index(
        "ix_question_submissions_status_created",
        "question_submissions",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_column("users", "is_moderator")
    op.drop_index("ix_question_submissions_status_created", table_name="question_submissions")
    op.drop_index(op.f("ix_question_submissions_status"), table_name="question_submissions")
    op.drop_index(op.f("ix_question_submissions_company_slug"), table_name="question_submissions")
    op.drop_index(op.f("ix_question_submissions_candidate_id"), table_name="question_submissions")
    op.drop_table("question_submissions")
