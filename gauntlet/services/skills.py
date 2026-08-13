"""Cross-interview skill graph persistence (spec sections 21-23, 30).

An interview's evidence is appended to ``skill_evidence`` and the candidate's
``candidate_skill_states`` are then recomputed *from the full evidence history*, not
patched incrementally. That matters: recency weighting means a concept's mastery can
change without new evidence, and a from-scratch recomputation is the only way the
stored value stays consistent with the model. It also means swapping the mastery
formula (for IRT, say) is a recompute, not a migration.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from gauntlet.content.taxonomy import ancestors_of, display_name
from gauntlet.db.models import CandidateSkillState, InterviewSession, SkillEvidence
from gauntlet.schemas import SkillReading
from gauntlet.skills.mastery import Evidence, compute_mastery, next_review

log = structlog.get_logger(__name__)


def merge_session_into_skill_graph(
    session: Session, record: InterviewSession, state: Mapping[str, Any]
) -> int:
    """Append this interview's evidence, then recompute the candidate's skill states."""
    rows = state.get("evidence") or []
    appended = 0

    for row in rows:
        observed_at = _parse_dt(row.get("observed_at")) or datetime.now(UTC)
        for concept_key in row.get("concept_keys", []):
            session.add(
                SkillEvidence(
                    candidate_id=record.candidate_id,
                    concept_key=concept_key,
                    session_id=record.id,
                    score=float(row.get("score", 0.0)),
                    difficulty=int(row.get("difficulty", 3)),
                    self_confidence=row.get("self_confidence"),
                    hints_used=int(row.get("hints_used", 0)),
                    is_followup=bool(row.get("is_followup", False)),
                    observed_at=observed_at,
                )
            )
            appended += 1

    session.flush()
    recompute_skill_states(session, record.candidate_id)
    log.info(
        "skills.merged",
        candidate=str(record.candidate_id),
        session=str(record.id),
        evidence_rows=appended,
    )
    return appended


def recompute_skill_states(session: Session, candidate_id: uuid.UUID) -> int:
    """Rebuild every skill state for a candidate from their full evidence history."""
    evidence_rows = session.scalars(
        select(SkillEvidence).where(SkillEvidence.candidate_id == candidate_id)
    ).all()

    by_concept: dict[str, list[Evidence]] = defaultdict(list)
    for evidence_row in evidence_rows:
        by_concept[evidence_row.concept_key].append(
            Evidence(
                score=float(evidence_row.score),
                difficulty=evidence_row.difficulty,
                observed_at=_ensure_tz(evidence_row.observed_at),
                self_confidence=evidence_row.self_confidence,
                hints_used=evidence_row.hints_used,
                is_followup=evidence_row.is_followup,
            )
        )

    existing = {
        row.concept_key: row
        for row in session.scalars(
            select(CandidateSkillState).where(CandidateSkillState.candidate_id == candidate_id)
        )
    }

    now = datetime.now(UTC)
    for concept_key, items in by_concept.items():
        state = compute_mastery(items, now)
        state_row = existing.get(concept_key)
        if state_row is None:
            state_row = CandidateSkillState(candidate_id=candidate_id, concept_key=concept_key)
            session.add(state_row)
            existing[concept_key] = state_row

        state_row.mastery = state.mastery
        state_row.confidence = state.confidence
        state_row.self_confidence = state.self_confidence
        state_row.evidence_count = state.evidence_count
        state_row.last_evidence_at = max(item.observed_at for item in items)

        due_at, interval = next_review(state.mastery, state_row.repetition_interval_days, now)
        state_row.due_at = due_at
        state_row.repetition_interval_days = interval

    session.flush()
    return len(by_concept)


def load_readings(session: Session, candidate_id: uuid.UUID) -> list[SkillReading]:
    """Persisted skill states as reading objects, strongest last."""
    rows = session.scalars(
        select(CandidateSkillState).where(CandidateSkillState.candidate_id == candidate_id)
    ).all()
    return sorted(
        (
            SkillReading(
                concept_key=row.concept_key,
                display_name=display_name(row.concept_key),
                mastery=float(row.mastery),
                confidence=float(row.confidence),
                evidence_count=row.evidence_count,
                self_confidence=float(row.self_confidence)
                if row.self_confidence is not None
                else None,
                is_misconception=False,
            )
            for row in rows
        ),
        key=lambda reading: reading.mastery,
    )


def rolled_up_readings(readings: list[SkillReading]) -> dict[str, SkillReading]:
    """Ancestor readings derived from leaf evidence, for the dashboard's headline bars."""
    buckets: dict[str, list[SkillReading]] = defaultdict(list)
    for reading in readings:
        for ancestor in ancestors_of(reading.concept_key):
            buckets[ancestor].append(reading)

    rolled: dict[str, SkillReading] = {}
    for key, group in buckets.items():
        total_weight = sum(max(item.confidence, 0.1) for item in group)
        mastery = (
            sum(item.mastery * max(item.confidence, 0.1) for item in group) / total_weight
            if total_weight
            else 0.0
        )
        rolled[key] = SkillReading(
            concept_key=key,
            display_name=display_name(key),
            mastery=mastery,
            confidence=total_weight / len(group) if group else 0.0,
            evidence_count=sum(item.evidence_count for item in group),
        )
    return rolled


def due_for_review(
    session: Session, candidate_id: uuid.UUID, limit: int = 10
) -> list[SkillReading]:
    """Concepts whose spaced-repetition interval has elapsed (spec section 30)."""
    now = datetime.now(UTC)
    rows = session.scalars(
        select(CandidateSkillState)
        .where(
            CandidateSkillState.candidate_id == candidate_id,
            CandidateSkillState.due_at.is_not(None),
            CandidateSkillState.due_at <= now,
        )
        .order_by(CandidateSkillState.mastery)
        .limit(limit)
    ).all()
    return [
        SkillReading(
            concept_key=row.concept_key,
            display_name=display_name(row.concept_key),
            mastery=float(row.mastery),
            confidence=float(row.confidence),
            evidence_count=row.evidence_count,
            self_confidence=float(row.self_confidence)
            if row.self_confidence is not None
            else None,
        )
        for row in rows
    ]


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return _ensure_tz(parsed)


def _ensure_tz(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)
