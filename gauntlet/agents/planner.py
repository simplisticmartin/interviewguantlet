"""Interview Planner Agent (spec section 4).

Produces the *opening hypothesis* for an interview: which areas, in what proportion,
starting at what difficulty. The adaptive router reshapes it from question two onwards,
so this is a starting distribution rather than a script.

Ordering matters here: the deterministic overlap between the job's demanded concepts and
the candidate's claimed concepts is computed first, in Python, and handed to the model as
context. The model shapes and justifies the plan; it does not get to hallucinate which
skills the candidate claims.
"""

from __future__ import annotations

from dataclasses import dataclass

from gauntlet.agents.base import Agent
from gauntlet.content.companies import CompanySeed
from gauntlet.content.taxonomy import get_concept, taxonomy_for_prompt
from gauntlet.prompts.catalog import INTERVIEW_PLANNER
from gauntlet.schemas import (
    InterviewMode,
    InterviewPlan,
    InterviewType,
    JobAnalysis,
    ResumeProfile,
)

# Roughly how long one question plus its follow-ups takes.
MINUTES_PER_QUESTION = 2.5

_LEVEL_DIFFICULTY: dict[str, int] = {
    "intern": 2,
    "junior": 2,
    "mid": 3,
    "senior": 4,
    "staff": 4,
    "principal": 5,
    "lead": 4,
}

# Modes that pin the interview to a single kind of question.
_MODE_LOCK: dict[InterviewMode, InterviewType] = {
    InterviewMode.CODING: InterviewType.DSA,
    InterviewMode.SYSTEM_DESIGN: InterviewType.SYSTEM_DESIGN,
    InterviewMode.BEHAVIORAL: InterviewType.BEHAVIORAL,
    InterviewMode.RESUME_DEFENSE: InterviewType.RESUME_DEFENSE,
}


@dataclass(frozen=True, slots=True)
class PlanRequest:
    profile: ResumeProfile
    job: JobAnalysis
    target_role: str
    target_level: str
    mode: InterviewMode
    minutes: int
    interview_types: list[InterviewType]
    company: CompanySeed | None = None
    weak_concepts: list[str] | None = None  # carried over from earlier interviews


class InterviewPlannerAgent(Agent):
    key = "planner"

    def build_plan(self, request: PlanRequest) -> InterviewPlan:
        overlap = compute_concept_overlap(request.profile, request.job)
        hints = build_focus_hints(request, overlap)
        target_count = question_budget(request.minutes)

        company_mix = request.company.interview_mix() if request.company else None

        claims = sorted(
            request.profile.claims, key=lambda claim: claim.probe_priority, reverse=True
        )
        claim_texts = [claim.claim_text for claim in claims[:5]]

        result = self.invoke(
            INTERVIEW_PLANNER,
            InterviewPlan,
            context={
                "target_role": request.target_role,
                "target_level": request.target_level,
                "mode": request.mode.value,
                "available_minutes": request.minutes,
                "target_question_count": target_count,
                "opening_difficulty": opening_difficulty(request.target_level),
                "allowed_interview_types": [t.value for t in request.interview_types],
                "focus_hints": hints,
                "concept_overlap": overlap,
                "candidate_claimed_concepts": request.profile.concept_keys,
                "job_weighted_concepts": [
                    {"concept_key": c.concept_key, "weight": c.weight}
                    for c in request.job.weighted_concepts
                ],
                "resume_claims_to_probe": claim_texts,
                "carried_over_weaknesses": request.weak_concepts or [],
                "company": (
                    {
                        "name": request.company.name,
                        "interview_mix": company_mix,
                    }
                    if request.company
                    else None
                ),
                "is_company_estimated": True,
                "taxonomy": taxonomy_for_prompt(),
            },
        )
        return _constrain(result.value, request, target_count)


def _as_float(value: object) -> float:
    """Narrow a heterogeneous JSON-ish value to a float for arithmetic."""
    return float(value) if isinstance(value, int | float | str) else 0.0


def opening_difficulty(level: str) -> int:
    return _LEVEL_DIFFICULTY.get(level.strip().lower(), 3)


def question_budget(minutes: int) -> int:
    from gauntlet.config import get_settings

    settings = get_settings()
    estimated = int(minutes / MINUTES_PER_QUESTION)
    return max(
        settings.min_questions_per_interview,
        min(settings.max_questions_per_interview, estimated),
    )


def compute_concept_overlap(profile: ResumeProfile, job: JobAnalysis) -> list[dict[str, object]]:
    """Concepts the job demands, scored by whether the candidate also claims them.

    The intersection is where a real loop probes hardest: the candidate invited the
    question by putting it on their resume, and the role requires it.
    """
    claimed = set(profile.concept_keys)
    for claim in profile.claims:
        claimed.update(claim.concept_keys)

    rows: list[dict[str, object]] = []
    for weighted in job.weighted_concepts:
        concept = get_concept(weighted.concept_key)
        if concept is None:
            continue
        is_claimed = weighted.concept_key in claimed
        rows.append(
            {
                "concept_key": weighted.concept_key,
                "display_name": concept.display_name,
                "interview_type": concept.interview_type.value,
                "job_weight": round(weighted.weight, 3),
                "claimed_by_candidate": is_claimed,
                # Claimed + demanded is the highest-value question in the loop.
                "priority": round(weighted.weight * (1.5 if is_claimed else 0.8), 3),
            }
        )

    # Concepts the candidate claims that the job did not mention still deserve some
    # coverage - resume defence works on exactly this territory.
    for key in sorted(claimed - {row["concept_key"] for row in rows}):
        concept = get_concept(key)
        if concept is None:
            continue
        rows.append(
            {
                "concept_key": key,
                "display_name": concept.display_name,
                "interview_type": concept.interview_type.value,
                "job_weight": 0.0,
                "claimed_by_candidate": True,
                "priority": 0.35,
            }
        )

    rows.sort(key=lambda row: _as_float(row.get("priority")), reverse=True)
    return rows[:30]


def build_focus_hints(
    request: PlanRequest, overlap: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Seed distribution over interview types, before the model refines it."""
    locked = _MODE_LOCK.get(request.mode)
    allowed = [locked] if locked else list(request.interview_types)
    if not allowed:
        allowed = [InterviewType.JAVA, InterviewType.SYSTEM_DESIGN, InterviewType.BEHAVIORAL]

    company_distribution: dict[str, float] = {}
    if request.company:
        mix = request.company.interview_mix()
        distribution = mix.get("distribution")
        if isinstance(distribution, dict):
            company_distribution = {str(k): float(v) for k, v in distribution.items()}

    by_type: dict[InterviewType, list[dict[str, object]]] = {}
    for row in overlap:
        try:
            interview_type = InterviewType(str(row["interview_type"]))
        except ValueError:
            continue
        if interview_type not in allowed:
            continue
        by_type.setdefault(interview_type, []).append(row)

    hints: list[dict[str, object]] = []
    for interview_type in allowed:
        rows = by_type.get(interview_type, [])
        evidence_weight = sum(_as_float(row.get("priority")) for row in rows)
        company_weight = company_distribution.get(interview_type.value, 0.0)
        # Blend what this candidate/job imply with the company's estimated shape.
        weight = 0.65 * evidence_weight + 0.35 * company_weight * 3.0
        if weight <= 0:
            weight = 0.15
        hints.append(
            {
                "interview_type": interview_type.value,
                "weight": round(weight, 3),
                "concept_keys": [str(row["concept_key"]) for row in rows[:8]],
                "rationale": (
                    f"{len(rows)} relevant concept(s) from the job/resume overlap"
                    + (
                        f"; company archetype weights this at {company_weight:.0%}"
                        if company_weight
                        else ""
                    )
                ),
            }
        )

    total = sum(_as_float(hint.get("weight")) for hint in hints)
    if total > 0:
        for hint in hints:
            hint["weight"] = round(_as_float(hint.get("weight")) / total, 3)
    return hints


def _constrain(plan: InterviewPlan, request: PlanRequest, target_count: int) -> InterviewPlan:
    """Keep the model inside the interview's actual constraints."""
    locked = _MODE_LOCK.get(request.mode)
    allowed = {locked} if locked else set(request.interview_types)

    areas = [area for area in plan.focus_areas if not allowed or area.interview_type in allowed]
    if not areas:
        areas = plan.focus_areas

    return plan.model_copy(
        update={
            "focus_areas": areas,
            "target_question_count": target_count,
            "opening_difficulty": max(1, min(5, plan.opening_difficulty)),
        }
    )
