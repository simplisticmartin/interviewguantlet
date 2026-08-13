"""Skills, analytics, and the study plan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import get_current_candidate, get_db
from apps.api.schemas import (
    AnalyticsResponse,
    SkillView,
    StudyPlanItemView,
    StudyPlanView,
)
from gauntlet.content.taxonomy import display_name
from gauntlet.db.models import (
    Candidate,
    CandidateSkillState,
    InterviewSession,
    Misconception,
    StudyPlan,
)
from gauntlet.services.skills import due_for_review
from gauntlet.skills.mastery import Calibration, classify_calibration

router = APIRouter(tags=["progress"])


@router.get("/skills", response_model=list[SkillView])
def get_skills(
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> list[SkillView]:
    rows = session.scalars(
        select(CandidateSkillState)
        .where(CandidateSkillState.candidate_id == candidate.id)
        .order_by(CandidateSkillState.mastery.desc())
    ).all()
    return [_skill_view(row) for row in rows]


@router.get("/skills/history")
def skill_history(
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    """Per-interview category scores, oldest first - the improvement curve."""
    rows = session.scalars(
        select(InterviewSession)
        .where(
            InterviewSession.candidate_id == candidate.id,
            InterviewSession.status == "completed",
        )
        .order_by(InterviewSession.ended_at)
    ).all()

    points = [
        {
            "session_id": str(row.id),
            "ended_at": row.ended_at.isoformat() if row.ended_at else None,
            "overall": (row.final_scorecard or {}).get("overall"),
            "categories": (row.final_scorecard or {}).get("category_scores", {}),
            "role": row.target_role,
        }
        for row in rows
    ]
    return {"points": points}


@router.get("/study-plan", response_model=StudyPlanView)
def get_study_plan(
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> StudyPlanView:
    plan = session.scalar(
        select(StudyPlan)
        .where(StudyPlan.candidate_id == candidate.id, StudyPlan.is_active.is_(True))
        .order_by(StudyPlan.created_at.desc())
    )
    if plan is None:
        return StudyPlanView(summary="", items=[])

    return StudyPlanView(
        id=plan.id,
        summary=plan.summary,
        created_at=plan.created_at,
        items=[
            StudyPlanItemView(
                priority=item.priority,
                concept_key=item.concept_key,
                display_name=display_name(item.concept_key),
                title=item.title,
                rationale=item.rationale,
                learn_items=list(item.learn_items),
                practice_items=list(item.practice_items),
                status=item.status,
            )
            for item in plan.items
        ],
    )


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> AnalyticsResponse:
    completed = session.scalars(
        select(InterviewSession).where(
            InterviewSession.candidate_id == candidate.id,
            InterviewSession.status == "completed",
        )
    ).all()

    overalls = [
        int((row.final_scorecard or {}).get("overall", 0))
        for row in completed
        if (row.final_scorecard or {}).get("overall") is not None
    ]

    skill_rows = session.scalars(
        select(CandidateSkillState).where(CandidateSkillState.candidate_id == candidate.id)
    ).all()
    views = [_skill_view(row) for row in skill_rows]

    misconceptions = session.scalars(
        select(Misconception)
        .where(Misconception.candidate_id == candidate.id, Misconception.status == "open")
        .order_by(Misconception.severity.desc(), Misconception.times_observed.desc())
    ).all()

    readiness = _readiness_bars(views)

    calibration_counts: dict[str, int] = {item.value: 0 for item in Calibration}
    for view in views:
        calibration_counts[view.calibration] = calibration_counts.get(view.calibration, 0) + 1

    return AnalyticsResponse(
        interviews_completed=len(completed),
        questions_answered=sum(len(row.questions) for row in completed),
        average_overall=round(sum(overalls) / len(overalls), 1) if overalls else None,
        readiness=readiness,
        strongest=sorted(
            (view for view in views if view.evidence_count),
            key=lambda view: view.mastery,
            reverse=True,
        )[:5],
        weakest=sorted(
            (view for view in views if view.evidence_count), key=lambda view: view.mastery
        )[:5],
        open_misconceptions=[
            {
                "concept_key": row.concept_key,
                "display_name": display_name(row.concept_key),
                "belief": row.belief,
                "correction": row.correction,
                "severity": row.severity,
                "times_observed": row.times_observed,
            }
            for row in misconceptions
        ],
        improvement=_improvement(completed),
        due_for_review=[
            SkillView(
                concept_key=reading.concept_key,
                display_name=reading.display_name,
                mastery=reading.mastery,
                confidence=reading.confidence,
                evidence_count=reading.evidence_count,
                self_confidence=reading.self_confidence,
                calibration=classify_calibration(
                    reading.mastery, reading.self_confidence
                ).value,
            )
            for reading in due_for_review(session, candidate.id)
        ],
        confidence_calibration=calibration_counts,
    )


def _skill_view(row: CandidateSkillState) -> SkillView:
    self_confidence = float(row.self_confidence) if row.self_confidence is not None else None
    return SkillView(
        concept_key=row.concept_key,
        display_name=display_name(row.concept_key),
        mastery=round(float(row.mastery), 4),
        confidence=round(float(row.confidence), 4),
        evidence_count=row.evidence_count,
        self_confidence=self_confidence,
        calibration=classify_calibration(float(row.mastery), self_confidence).value,
        due_at=row.due_at,
    )


def _readiness_bars(views: list[SkillView]) -> list[SkillView]:
    """Top-level domain rollups for the dashboard bars."""
    from collections import defaultdict

    buckets: dict[str, list[SkillView]] = defaultdict(list)
    for view in views:
        if view.evidence_count:
            buckets[view.concept_key.split(".")[0]].append(view)

    bars: list[SkillView] = []
    for root, group in buckets.items():
        total = sum(max(item.confidence, 0.1) for item in group)
        mastery = sum(item.mastery * max(item.confidence, 0.1) for item in group) / total
        bars.append(
            SkillView(
                concept_key=root,
                display_name=display_name(root),
                mastery=round(mastery, 4),
                confidence=round(total / len(group), 4),
                evidence_count=sum(item.evidence_count for item in group),
            )
        )
    return sorted(bars, key=lambda item: item.mastery, reverse=True)


def _improvement(sessions: Sequence[InterviewSession]) -> list[dict[str, Any]]:
    ordered = sorted(sessions, key=lambda row: row.ended_at or row.created_at)
    return [
        {
            "session_id": str(row.id),
            "label": (row.ended_at or row.created_at).strftime("%d %b"),
            "overall": (row.final_scorecard or {}).get("overall", 0),
            "role": row.target_role,
        }
        for row in ordered
    ]
