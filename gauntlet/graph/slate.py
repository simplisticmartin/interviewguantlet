"""Turning a plan's weighted distribution into a concrete ordered slate.

The planner agent decides *proportions* ("30% distributed systems, 25% design").
This module turns those proportions into an actual ordered list of slots, because a
distribution cannot be asked as a question.

The slate is a starting order, not a script. The adaptive router overrides the next
slot whenever evidence says to go deeper, back off, or probe - the slate is what the
interview falls back to when there is no reason to deviate.

Allocation uses the largest-remainder method so proportions round to exact totals
without a topic silently losing its only slot.
"""

from __future__ import annotations

from typing import Any

from gauntlet.content.taxonomy import concepts_for_type, examinable_under, get_concept, is_branch
from gauntlet.evaluation.rubrics import rubric_for_concept, rubric_index
from gauntlet.schemas import InterviewPlan, InterviewType, ResumeClaimModel

# Fraction of slots reserved for resume cross-examination when the resume has
# high-priority claims and the interview is long enough to afford it.
RESUME_SLOT_RATIO = 0.25
MAX_RESUME_SLOTS = 3
# Below this, a claim is not specific enough to cross-examine. "8 years of experience"
# is a fact about a calendar, not a claim with depth behind it - probing it wastes a slot.
MIN_PROBE_PRIORITY = 4


def allocate(weights: dict[InterviewType, float], total: int) -> dict[InterviewType, int]:
    """Largest-remainder allocation of ``total`` slots across weighted types.

    With one guarantee on top of the pure maths: an area the planner put in the plan
    gets at least one question. Rounding a 10% area down to zero in a short interview
    silently deletes a topic the planner deliberately chose to cover. When there are
    more areas than slots, the highest-weighted areas win and the rest are dropped
    explicitly rather than by rounding accident.
    """
    if total <= 0 or not weights:
        return {}

    by_weight = sorted(weights.items(), key=lambda pair: pair[1], reverse=True)
    if len(by_weight) > total:
        by_weight = by_weight[:total]
    eligible = dict(by_weight)
    weight_total = sum(eligible.values()) or 1.0

    raw = {key: (weight / weight_total) * total for key, weight in eligible.items()}
    allocated = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(allocated.values())

    remainders = sorted(
        ((key, raw[key] - allocated[key]) for key in raw),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for index in range(remaining):
        allocated[remainders[index % len(remainders)][0]] += 1

    # Guarantee coverage: give a slot to each starved area, taken from the largest.
    for key, count in allocated.items():
        if count > 0:
            continue
        donor = max(allocated, key=lambda item: allocated[item])
        if allocated[donor] <= 1:
            break
        allocated[donor] -= 1
        allocated[key] = 1

    return {key: count for key, count in allocated.items() if count > 0}


def build_slate(
    plan: InterviewPlan,
    claims: list[ResumeClaimModel] | None = None,
    opening_difficulty: int = 3,
) -> list[dict[str, Any]]:
    """Materialise the plan into ordered question slots."""
    total = plan.target_question_count
    claims = sorted(
        (claim for claim in (claims or []) if claim.probe_priority >= MIN_PROBE_PRIORITY),
        key=lambda claim: claim.probe_priority,
        reverse=True,
    )

    resume_slots = 0
    if claims and total >= 4:
        resume_slots = min(MAX_RESUME_SLOTS, len(claims), max(1, round(total * RESUME_SLOT_RATIO)))

    technical_total = max(1, total - resume_slots)
    allocation = allocate(plan.normalised_weights(), technical_total)

    per_type: dict[InterviewType, list[dict[str, Any]]] = {}
    for area in plan.focus_areas:
        count = allocation.get(area.interview_type, 0)
        if count <= 0:
            continue
        per_type[area.interview_type] = _slots_for_area(
            area.interview_type, area.concept_keys, count, opening_difficulty
        )

    # Interleave types round-robin so the interview does not spend ten minutes on one
    # subject before touching another - real loops move between areas.
    slate: list[dict[str, Any]] = []
    cursor = 0
    while any(len(slots) > cursor for slots in per_type.values()):
        for interview_type in per_type:
            slots = per_type[interview_type]
            if cursor < len(slots):
                slate.append(slots[cursor])
        cursor += 1

    for index, claim in enumerate(claims[:resume_slots]):
        slot = _resume_slot(claim, opening_difficulty)
        # Space resume probes through the interview rather than front-loading them.
        position = min(len(slate), 2 + index * 3)
        slate.insert(position, slot)

    return slate[:total]


def resolve_examinable(concept_keys: list[str], difficulty: int) -> list[str]:
    """Expand branch concepts into askable ones, best rubric coverage first.

    The planner reasons about areas ("Kafka", "Spring"), but you cannot ask a question
    about a category. Concepts that carry an authored rubric come first, because those
    are the ones the evaluator can grade against something specific rather than the
    generic fallback.
    """
    authored = rubric_index()
    resolved: list[str] = []

    for key in concept_keys:
        if get_concept(key) is None:
            continue
        if not is_branch(key):
            resolved.append(key)
            continue

        candidates = examinable_under(key, difficulty)
        # A branch with its own authored rubric is still askable: "walk me through your
        # caching design" is a real question even though caching has sub-concepts.
        if key in authored:
            candidates.append(key)
        # Stable ordering: rubric-backed concepts first, then by key for determinism.
        candidates.sort(key=lambda item: (item not in authored, item))
        resolved.extend(candidates)

    seen: list[str] = []
    for key in resolved:
        if key not in seen:
            seen.append(key)
    return seen


def _slots_for_area(
    interview_type: InterviewType,
    concept_keys: list[str],
    count: int,
    opening_difficulty: int,
) -> list[dict[str, Any]]:
    keys = resolve_examinable(concept_keys, opening_difficulty)
    if not keys:
        # The plan named no usable concept for this area: fall back to the taxonomy,
        # preferring concepts whose difficulty band contains the opening difficulty.
        keys = [
            concept.key
            for concept in concepts_for_type(interview_type)
            if concept.difficulty_floor <= opening_difficulty <= concept.difficulty_ceiling
        ] or [concept.key for concept in concepts_for_type(interview_type)]

    slots: list[dict[str, Any]] = []
    for index in range(count):
        key = keys[index % len(keys)] if keys else ""
        concept = get_concept(key)
        difficulty = opening_difficulty
        if concept is not None:
            difficulty = max(concept.difficulty_floor, min(concept.difficulty_ceiling, difficulty))
        rubric = rubric_for_concept(key, interview_type) if key else None
        slots.append(
            {
                "interview_type": interview_type.value,
                "concept_keys": [key] if key else [],
                "difficulty": difficulty,
                "rubric_key": rubric.key if rubric else None,
                "is_resume_probe": False,
                "claim_text": None,
                # Ask for self-rated confidence on the first slot of each area so the
                # calibration signal is spread across topics (spec section 22).
                "ask_confidence": index == 0,
            }
        )
    return slots


def _resume_slot(claim: ResumeClaimModel, opening_difficulty: int) -> dict[str, Any]:
    examinable = resolve_examinable(claim.concept_keys, opening_difficulty)
    concept_key = examinable[0] if examinable else ""
    concept = get_concept(concept_key)
    interview_type = (
        concept.interview_type if concept else InterviewType.RESUME_DEFENSE
    )
    return {
        "interview_type": interview_type.value,
        "concept_keys": examinable[:3],
        "difficulty": max(2, min(5, opening_difficulty)),
        "rubric_key": "resume.claim_defense",
        "is_resume_probe": True,
        "claim_text": claim.claim_text,
        "claim_has_metric": claim.has_metric,
        "ask_confidence": False,
    }
