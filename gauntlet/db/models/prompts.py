"""Versioned prompts and rubrics (spec sections 18 and 42).

Prompts live in code as the source of truth (``gauntlet/prompts``) and are mirrored
into these tables on startup. Evaluations then reference a concrete row, so any past
verdict can be traced to the exact prompt text and model that produced it.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from gauntlet.db.base import Base, TimestampMixin


class PromptVersion(Base, TimestampMixin):
    __tablename__ = "prompt_versions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(120))
    temperature: Mapped[float | None] = mapped_column(Numeric(3, 2))
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_prompt_name_version"),)


class Rubric(Base, TimestampMixin):
    """A scoring rubric: named dimensions with weights, not a vibes-based 1-10."""

    __tablename__ = "rubrics"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    concept_key: Mapped[str | None] = mapped_column(String(200), index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    # [{"key": "treeification", "label": "...", "weight": 1.0, "hint": "..."}]
    dimensions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (UniqueConstraint("key", "version", name="uq_rubric_key_version"),)
