"""Study planner (spec sections 29-30).

Converts measured gaps into a plan the candidate can act on today. Two design rules:

* Priority follows evidence, not topic popularity: confidently-wrong beliefs first,
  then role-critical low scores, then known gaps, then confidence deficits.
* Re-attempt prompts are *reworded*, not repeated. If the candidate is re-asked the exact
  question they failed, they will memorise that answer instead of learning the concept,
  which is the failure mode spaced repetition exists to avoid. We pull a different
  question on the same concept out of the corpus to serve as the re-attempt.
"""

from __future__ import annotations

import structlog

from gauntlet.agents.base import Agent
from gauntlet.content.taxonomy import children_of, deeper_concepts, display_name
from gauntlet.llm.base import StructuredOutputError
from gauntlet.prompts.catalog import STUDY_PLANNER
from gauntlet.retrieval.question_index import get_question_index
from gauntlet.schemas import (
    MisconceptionFinding,
    SkillReading,
    StudyPlanItemModel,
    StudyPlanModel,
)
from gauntlet.skills.mastery import Calibration, classify_calibration

log = structlog.get_logger(__name__)

MAX_ITEMS = 6


class StudyPlannerAgent(Agent):
    key = "study_planner"

    def build(
        self,
        *,
        readings: list[SkillReading],
        misconceptions: list[MisconceptionFinding],
        job_concept_keys: list[str],
        asked_prompts: list[str],
        target_role: str = "Software Engineer",
        target_level: str = "senior",
    ) -> StudyPlanModel:
        detected = [finding for finding in misconceptions if finding.detected]
        weaknesses = _rank_weaknesses(readings, job_concept_keys)

        try:
            result = self.invoke(
                STUDY_PLANNER,
                StudyPlanModel,
                context={
                    "target_role": target_role,
                    "target_level": target_level,
                    "max_items": MAX_ITEMS,
                    "misconceptions": [
                        {
                            **finding.model_dump(),
                            "reattempt_prompt": _reattempt_prompt(
                                finding.concept_key, asked_prompts
                            ),
                        }
                        for finding in detected
                    ],
                    "weaknesses": [
                        {
                            "concept_key": reading.concept_key,
                            "display_name": reading.display_name,
                            "mastery": round(reading.mastery, 3),
                            "confidence": round(reading.confidence, 3),
                            "self_confidence": reading.self_confidence,
                            "evidence_count": reading.evidence_count,
                            "calibration": classify_calibration(
                                reading.mastery, reading.self_confidence
                            ).value,
                            "job_relevant": reading.concept_key in set(job_concept_keys),
                            "sub_concepts": _sub_concepts(reading.concept_key),
                            "practice_prompts": _practice_prompts(
                                reading.concept_key, asked_prompts
                            ),
                            "reattempt_prompt": _reattempt_prompt(
                                reading.concept_key, asked_prompts
                            ),
                        }
                        for reading in weaknesses
                    ],
                },
            )
        except StructuredOutputError:
            log.warning("study_planner.failed")
            return _fallback_plan(detected, weaknesses, asked_prompts)

        plan = result.value
        items = plan.items[:MAX_ITEMS]
        # Renumber so priorities are always 1..n and strictly ordered.
        items = [
            item.model_copy(update={"priority": index})
            for index, item in enumerate(
                sorted(items, key=lambda item: item.priority), start=1
            )
        ]
        return plan.model_copy(update={"items": items})


def _rank_weaknesses(
    readings: list[SkillReading], job_concept_keys: list[str]
) -> list[SkillReading]:
    """Low mastery matters more when the target role demands the concept."""
    job_keys = set(job_concept_keys)
    scored = [
        (
            reading,
            (1.0 - reading.mastery)
            * (1.5 if reading.concept_key in job_keys else 1.0)
            * (0.6 + 0.4 * reading.confidence),
        )
        for reading in readings
        if reading.evidence_count
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [reading for reading, _ in scored[:MAX_ITEMS]]


def _sub_concepts(concept_key: str) -> list[str]:
    """Concrete things to learn: children first, then the interviewer's descent path."""
    names = [child.display_name for child in children_of(concept_key)]
    names.extend(display_name(key) for key in deeper_concepts(concept_key))
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen[:6]


def _practice_prompts(concept_key: str, asked_prompts: list[str]) -> list[str]:
    already = set(asked_prompts)
    found = get_question_index().for_concepts(
        concept_keys=[concept_key], difficulty=3, limit=6
    )
    return [item.seed.question for item in found if item.seed.question not in already][:3]


def _reattempt_prompt(concept_key: str | None, asked_prompts: list[str]) -> str | None:
    """A *different* question on the same concept (spec section 30)."""
    if not concept_key:
        return None
    prompts = _practice_prompts(concept_key, asked_prompts)
    return prompts[0] if prompts else None


def _fallback_plan(
    misconceptions: list[MisconceptionFinding],
    weaknesses: list[SkillReading],
    asked_prompts: list[str],
) -> StudyPlanModel:
    items: list[StudyPlanItemModel] = []
    priority = 1

    for finding in misconceptions[:3]:
        concept_key = finding.concept_key or "general"
        items.append(
            StudyPlanItemModel(
                priority=priority,
                concept_key=concept_key,
                title=f"Correct your model of {display_name(concept_key)}",
                rationale=(
                    f'You stated during the interview: "{finding.belief}". '
                    f"That is inaccurate: {finding.correction}"
                ),
                learn_items=[finding.correction, *_sub_concepts(concept_key)[:3]],
                practice_items=[
                    {"type": "question", "prompt": prompt}
                    for prompt in _practice_prompts(concept_key, asked_prompts)
                ],
                reattempt_prompt=_reattempt_prompt(concept_key, asked_prompts),
            )
        )
        priority += 1

    misconception_keys = {f.concept_key for f in misconceptions}
    for reading in weaknesses:
        if reading.concept_key in misconception_keys or priority > MAX_ITEMS:
            continue
        calibration = classify_calibration(reading.mastery, reading.self_confidence)
        if calibration is Calibration.CONFIDENCE_DEFICIT:
            rationale = (
                f"You scored {reading.mastery:.0%} on {reading.display_name} but rated "
                "your own confidence low. This needs rehearsal under pressure, not study."
            )
        else:
            rationale = (
                f"Measured mastery {reading.mastery:.0%} across "
                f"{reading.evidence_count} question(s) in this interview."
            )
        items.append(
            StudyPlanItemModel(
                priority=priority,
                concept_key=reading.concept_key,
                title=f"Build depth in {reading.display_name}",
                rationale=rationale,
                learn_items=_sub_concepts(reading.concept_key),
                practice_items=[
                    {"type": "question", "prompt": prompt}
                    for prompt in _practice_prompts(reading.concept_key, asked_prompts)
                ],
                reattempt_prompt=_reattempt_prompt(reading.concept_key, asked_prompts),
            )
        )
        priority += 1

    return StudyPlanModel(
        summary=(
            f"{len(items)} item(s), ordered by interview impact: confidently-wrong "
            "beliefs first, then role-critical gaps."
        ),
        items=items,
    )
