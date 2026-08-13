"""Adaptive router (spec section 2).

Decides where the interview goes after every answer. The LLM proposes; deterministic
rules dispose. Hard interview economics - don't burn three questions on a concept
already scored 0.9, don't chain more than two follow-ups, don't descend into a concept
that isn't in the taxonomy - are enforced in code, because a model that occasionally
ignores them would waste the candidate's limited interview time.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from gauntlet.agents.base import Agent
from gauntlet.content.taxonomy import concept_index, display_name
from gauntlet.llm.base import StructuredOutputError
from gauntlet.prompts.catalog import ADAPTIVE_ROUTER
from gauntlet.schemas import AdaptiveDecision, AdaptiveDirection, AggregateEvaluation
from gauntlet.skills.graph import SkillGraph

log = structlog.get_logger(__name__)

MAX_FOLLOWUPS_PER_CONCEPT = 2
SATURATION_MASTERY = 0.85
STRONG_SCORE = 0.8
WEAK_SCORE = 0.35


@dataclass(frozen=True, slots=True)
class RoutingContext:
    concept_key: str
    evaluation: AggregateEvaluation
    skill_graph: SkillGraph
    followups_on_concept: int
    questions_asked: int
    questions_remaining: int
    minutes_remaining: float
    available_concepts: list[str]


class AdaptiveRouterAgent(Agent):
    key = "router"

    def decide(self, context: RoutingContext) -> AdaptiveDecision:
        deeper = [
            key
            for key in _deeper_candidates(context)
            if context.skill_graph.evidence_count(key) == 0
        ]

        forced = self._forced_decision(context, deeper)
        if forced is not None:
            log.info("router.forced", direction=forced.direction.value, reason=forced.reason)
            return forced

        try:
            result = self.invoke(
                ADAPTIVE_ROUTER,
                AdaptiveDecision,
                context={
                    "concept_key": context.concept_key,
                    "concept_display_name": display_name(context.concept_key),
                    "last_evaluation": {
                        "score": round(context.evaluation.score, 3),
                        "confidence": round(context.evaluation.confidence, 3),
                        "demonstrated": context.evaluation.demonstrated,
                        "missing": context.evaluation.missing,
                        "incorrect": context.evaluation.incorrect,
                        "misconception_detected": context.evaluation.misconception.detected,
                        "judge_disagreement": context.evaluation.disagreement,
                    },
                    "followups_on_concept": context.followups_on_concept,
                    "questions_asked": context.questions_asked,
                    "questions_remaining": context.questions_remaining,
                    "minutes_remaining": round(context.minutes_remaining, 1),
                    "deeper_concepts": deeper,
                    "available_concepts": context.available_concepts,
                    "skill_snapshot": {
                        reading.concept_key: round(reading.mastery, 2)
                        for reading in context.skill_graph.leaf_readings()
                    },
                },
            )
            decision = result.value
        except StructuredOutputError:
            log.warning("router.llm_failed", concept=context.concept_key)
            return self._fallback(context, deeper)

        return self._constrain(decision, context, deeper)

    # --- Deterministic guards -------------------------------------------

    def _forced_decision(
        self, context: RoutingContext, deeper: list[str]
    ) -> AdaptiveDecision | None:
        """Situations where there is nothing for the model to decide."""
        if context.questions_remaining <= 0 or context.minutes_remaining <= 0.5:
            return AdaptiveDecision(
                direction=AdaptiveDirection.MOVE_ON,
                reason="Interview budget exhausted.",
            )
        if context.followups_on_concept >= MAX_FOLLOWUPS_PER_CONCEPT:
            return AdaptiveDecision(
                direction=AdaptiveDirection.LATERAL,
                reason=(
                    f"Already asked {context.followups_on_concept} follow-ups on "
                    f"{context.concept_key}; further probing would not add signal."
                ),
            )
        if (
            context.evaluation.misconception.detected
            and context.followups_on_concept < MAX_FOLLOWUPS_PER_CONCEPT
        ):
            # Confidently-wrong is the single highest-value finding: always chase it.
            return AdaptiveDecision(
                direction=AdaptiveDirection.PROBE,
                next_concept_key=context.concept_key,
                reason="Misconception detected; probing before scoring it as understood.",
            )
        if context.skill_graph.is_saturated(context.concept_key, SATURATION_MASTERY):
            return AdaptiveDecision(
                direction=AdaptiveDirection.HARDER if deeper else AdaptiveDirection.LATERAL,
                next_concept_key=deeper[0] if deeper else None,
                reason=f"{display_name(context.concept_key)} is already well evidenced.",
                difficulty_delta=1,
            )
        return None

    def _constrain(
        self, decision: AdaptiveDecision, context: RoutingContext, deeper: list[str]
    ) -> AdaptiveDecision:
        index = concept_index()
        next_key = decision.next_concept_key

        if next_key and next_key not in index:
            log.warning("router.unknown_concept", proposed=next_key)
            next_key = None

        if decision.direction is AdaptiveDirection.DEEPER:
            if not next_key or next_key == context.concept_key:
                next_key = deeper[0] if deeper else None
            if not next_key:
                return AdaptiveDecision(
                    direction=AdaptiveDirection.HARDER,
                    reason="No unexplored sub-concept remains; raising difficulty instead.",
                    difficulty_delta=1,
                )

        if decision.direction is AdaptiveDirection.PROBE and (
            context.followups_on_concept >= MAX_FOLLOWUPS_PER_CONCEPT
        ):
            return AdaptiveDecision(
                direction=AdaptiveDirection.LATERAL,
                reason="Follow-up budget for this concept is spent.",
            )

        return decision.model_copy(update={"next_concept_key": next_key})

    def _fallback(self, context: RoutingContext, deeper: list[str]) -> AdaptiveDecision:
        """Pure-rule routing when the model is unavailable."""
        score = context.evaluation.score
        if score >= STRONG_SCORE and deeper:
            return AdaptiveDecision(
                direction=AdaptiveDirection.DEEPER,
                next_concept_key=deeper[0],
                reason=f"Strong answer ({score:.2f}); descending.",
                difficulty_delta=1,
            )
        if score >= STRONG_SCORE:
            return AdaptiveDecision(
                direction=AdaptiveDirection.HARDER,
                reason=f"Strong answer ({score:.2f}); raising difficulty.",
                difficulty_delta=1,
            )
        if score < WEAK_SCORE:
            return AdaptiveDecision(
                direction=AdaptiveDirection.EASIER,
                reason=f"Weak answer ({score:.2f}); finding the floor.",
                difficulty_delta=-1,
            )
        return AdaptiveDecision(
            direction=AdaptiveDirection.LATERAL, reason=f"Adequate answer ({score:.2f})."
        )


def _deeper_candidates(context: RoutingContext) -> list[str]:
    from gauntlet.content.taxonomy import deeper_concepts

    return deeper_concepts(context.concept_key)
