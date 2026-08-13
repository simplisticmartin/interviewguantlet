"""Hiring committee (spec section 27).

The committee aggregates; it does not re-interview. Its inputs are the readings the
skill graph produced and the evidence the judges quoted, so every line of the verdict
traces back to something the candidate actually said.

The committee is also where the product's honesty constraint is enforced: it produces a
*simulated* recommendation, never a prediction about a real company's decision.
"""

from __future__ import annotations

import structlog

from gauntlet.agents.base import Agent
from gauntlet.llm.base import StructuredOutputError
from gauntlet.prompts.catalog import HIRING_COMMITTEE
from gauntlet.schemas import CommitteeVerdict, MisconceptionFinding, SkillReading

log = structlog.get_logger(__name__)

VALID_RECOMMENDATIONS = frozenset(
    {"STRONG_HIRE", "HIRE", "LEAN_HIRE", "LEAN_NO_HIRE", "NO_HIRE", "NO_DECISION"}
)


class HiringCommitteeAgent(Agent):
    key = "hiring_committee"

    def deliberate(
        self,
        *,
        readings: list[SkillReading],
        category_scores: dict[str, int],
        misconceptions: list[MisconceptionFinding],
        evidence_quotes: list[str],
        target_role: str,
        target_level: str,
        company_name: str | None = None,
        questions_asked: int = 0,
    ) -> CommitteeVerdict:
        strengths = sorted(
            (r for r in readings if r.evidence_count), key=lambda r: r.mastery, reverse=True
        )[:5]
        weaknesses = sorted((r for r in readings if r.evidence_count), key=lambda r: r.mastery)[:5]

        try:
            result = self.invoke(
                HIRING_COMMITTEE,
                CommitteeVerdict,
                context={
                    "target_role": target_role,
                    "target_level": target_level,
                    "company": company_name,
                    "simulation_disclaimer": (
                        "This is a Gauntlet simulation. Do not claim to predict the "
                        "company's real decision or describe their actual process."
                    ),
                    "questions_asked": questions_asked,
                    "category_scores": {
                        key: round(value / 100, 3) for key, value in category_scores.items()
                    },
                    "strengths": [reading.model_dump() for reading in strengths],
                    "weaknesses": [reading.model_dump() for reading in weaknesses],
                    "misconceptions": [
                        finding.model_dump() for finding in misconceptions if finding.detected
                    ],
                    "evidence_quotes": evidence_quotes[:20],
                },
            )
        except StructuredOutputError:
            log.warning("committee.failed", role=target_role)
            return _fallback_verdict(category_scores, strengths, weaknesses, misconceptions)

        verdict = result.value
        if verdict.recommendation not in VALID_RECOMMENDATIONS:
            verdict = verdict.model_copy(update={"recommendation": "NO_DECISION"})
        if questions_asked < 3:
            # Too little evidence to make a call, whatever the model wants to say.
            verdict = verdict.model_copy(
                update={
                    "recommendation": "NO_DECISION",
                    "risks": [
                        *verdict.risks,
                        f"Only {questions_asked} question(s) answered - insufficient "
                        "evidence for a recommendation.",
                    ],
                }
            )
        return verdict


def _fallback_verdict(
    category_scores: dict[str, int],
    strengths: list[SkillReading],
    weaknesses: list[SkillReading],
    misconceptions: list[MisconceptionFinding],
) -> CommitteeVerdict:
    overall = (
        (sum(category_scores.values()) / len(category_scores) / 100) if category_scores else 0.0
    )
    if overall >= 0.85:
        recommendation = "STRONG_HIRE"
    elif overall >= 0.72:
        recommendation = "HIRE"
    elif overall >= 0.6:
        recommendation = "LEAN_HIRE"
    elif overall >= 0.45:
        recommendation = "LEAN_NO_HIRE"
    else:
        recommendation = "NO_HIRE"

    detected = [finding for finding in misconceptions if finding.detected]
    if detected and recommendation in {"STRONG_HIRE", "HIRE"}:
        recommendation = "LEAN_HIRE"

    return CommitteeVerdict(
        recommendation=recommendation,
        scores={key: round(value / 100, 3) for key, value in category_scores.items()},
        strengths=[
            f"{reading.display_name}: mastery {reading.mastery:.2f} "
            f"({reading.evidence_count} question(s))"
            for reading in strengths
        ],
        risks=[
            f"{reading.display_name}: mastery {reading.mastery:.2f}" for reading in weaknesses
        ],
        evidence=[finding.evidence_quote or finding.belief for finding in detected],
        next_steps=[f"Study {reading.display_name}" for reading in weaknesses[:3]],
        most_likely_rejection_reason=(
            f"Confidently incorrect about {detected[0].concept_key}: {detected[0].belief}"
            if detected
            else (
                f"Insufficient depth in {weaknesses[0].display_name} for the target level."
                if weaknesses
                else ""
            )
        ),
    )
