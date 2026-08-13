"""Intake nodes: parse the candidate, analyse the job, retrieve company patterns, plan."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from gauntlet.agents.intake import JobAnalyzerAgent, ResumeParserAgent
from gauntlet.agents.planner import InterviewPlannerAgent, PlanRequest
from gauntlet.content.companies import find_company
from gauntlet.graph.slate import build_slate
from gauntlet.graph.state import InterviewState, interview_types_of, mode_of
from gauntlet.schemas import InterviewPlan, InterviewType, JobAnalysis, ResumeProfile

log = structlog.get_logger(__name__)


def parse_candidate(state: InterviewState) -> dict[str, Any]:
    """Extract a structured profile and probe-worthy claims from the resume."""
    resume_text = state.get("resume_text", "")
    if not resume_text.strip():
        return {
            "resume_profile": ResumeProfile().model_dump(mode="json"),
            "interviewer_notes": ["No resume supplied - resume cross-examination skipped."],
        }

    profile = ResumeParserAgent().parse(resume_text)
    log.info(
        "graph.parse_candidate",
        session=state.get("session_id"),
        concepts=len(profile.concept_keys),
        claims=len(profile.claims),
    )
    return {
        "resume_profile": profile.model_dump(mode="json"),
        "interviewer_notes": [
            f"Resume parsed: {len(profile.concept_keys)} known concepts, "
            f"{len(profile.claims)} checkable claims."
        ],
    }


def analyze_job(state: InterviewState) -> dict[str, Any]:
    """Turn the job description into weighted, assessable concepts."""
    job_text = state.get("job_text", "")
    if not job_text.strip():
        return {
            "job_description": JobAnalysis(
                title=state.get("target_role", "Software Engineer"),
                level=state.get("target_level", "senior"),
                summary="No job description supplied; using role and level only.",
            ).model_dump(mode="json"),
            "interviewer_notes": ["No job description supplied - planning from role/level."],
        }

    analysis = JobAnalyzerAgent().analyze(job_text)
    log.info(
        "graph.analyze_job",
        session=state.get("session_id"),
        concepts=len(analysis.weighted_concepts),
    )
    return {
        "job_description": analysis.model_dump(mode="json"),
        "interviewer_notes": [
            f"Job analysed: {len(analysis.weighted_concepts)} assessable concepts."
        ],
    }


def retrieve_company_patterns(state: InterviewState) -> dict[str, Any]:
    """Look up what we believe about this company's interview shape.

    Gauntlet ships only archetype-based *estimates*; the payload carries the evidence
    label so every downstream consumer - planner, UI, report - states which it is
    (spec sections 9 and 26).
    """
    slug = state.get("target_company")
    if not slug:
        return {"company_patterns": {"known": False, "evidence": "none"}}

    company = find_company(slug)
    if company is None:
        return {
            "company_patterns": {"known": False, "evidence": "none", "requested": slug},
            "interviewer_notes": [f"Company '{slug}' not in catalogue - planning generically."],
        }

    mix = company.interview_mix()
    return {
        "company_patterns": {
            "known": True,
            "slug": company.slug,
            "name": company.name,
            "sector": company.sector,
            **mix,
        },
        "interviewer_notes": [
            f"Company profile loaded for {company.name} (evidence: {mix['evidence']})."
        ],
    }


def build_interview_plan(state: InterviewState) -> dict[str, Any]:
    """Produce the opening distribution and materialise it into an ordered slate."""
    profile = ResumeProfile.model_validate(state.get("resume_profile") or {})
    job = JobAnalysis.model_validate(
        state.get("job_description")
        or {"title": state.get("target_role", ""), "level": state.get("target_level", "senior")}
    )

    patterns = state.get("company_patterns") or {}
    company = find_company(patterns["slug"]) if patterns.get("known") else None

    types = interview_types_of(state) or [
        InterviewType.JAVA,
        InterviewType.SYSTEM_DESIGN,
        InterviewType.BEHAVIORAL,
    ]

    plan = InterviewPlannerAgent().build_plan(
        PlanRequest(
            profile=profile,
            job=job,
            target_role=state.get("target_role", "Software Engineer"),
            target_level=state.get("target_level", "senior"),
            mode=mode_of(state),
            minutes=int(state.get("planned_minutes", 20)),
            interview_types=types,
            company=company,
        )
    )

    slate = build_slate(plan, profile.claims, plan.opening_difficulty)
    payload = plan.model_dump(mode="json")
    payload["slate"] = slate
    payload["company_evidence"] = patterns.get("evidence", "none")

    log.info(
        "graph.build_plan",
        session=state.get("session_id"),
        areas=len(plan.focus_areas),
        slots=len(slate),
        opening_difficulty=plan.opening_difficulty,
    )

    return {
        "interview_plan": payload,
        "difficulty": plan.opening_difficulty,
        "plan_cursor": 0,
        "status": "in_progress",
        "started_at": datetime.now(UTC).isoformat(),
        "interviewer_notes": [
            f"Plan: {len(slate)} slots across "
            f"{', '.join(area.interview_type.value for area in plan.focus_areas)}."
        ],
    }


def plan_summary(plan: InterviewPlan) -> str:
    parts = [
        f"{area.interview_type.value} {weight:.0%}"
        for area, weight in zip(
            plan.focus_areas, plan.normalised_weights().values(), strict=False
        )
    ]
    return ", ".join(parts)
