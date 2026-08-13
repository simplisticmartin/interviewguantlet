"""Adaptive routing and the stop condition."""

from __future__ import annotations

from typing import Any

import structlog

from gauntlet.agents.router import AdaptiveRouterAgent, RoutingContext
from gauntlet.config import get_settings
from gauntlet.graph.state import (
    InterviewState,
    minutes_remaining,
    questions_asked,
    questions_remaining,
    skill_graph_from_state,
)
from gauntlet.schemas import AggregateEvaluation

log = structlog.get_logger(__name__)


def adaptive_router(state: InterviewState) -> dict[str, Any]:
    """Decide the next move from the last answer and the running skill picture."""
    evaluation = AggregateEvaluation.model_validate(state.get("last_evaluation") or {"score": 0.0})
    graph = skill_graph_from_state(state)
    concept_key = state.get("current_concept_key") or ""

    plan = state.get("interview_plan") or {}
    available = sorted(
        {
            key
            for slot in plan.get("slate", [])
            for key in slot.get("concept_keys", [])
        }
        | set(graph.observed_keys())
    )

    decision = AdaptiveRouterAgent().decide(
        RoutingContext(
            concept_key=concept_key,
            evaluation=evaluation,
            skill_graph=graph,
            followups_on_concept=int(state.get("followups_on_concept", 0)),
            questions_asked=questions_asked(state),
            questions_remaining=questions_remaining(state),
            minutes_remaining=minutes_remaining(state),
            available_concepts=available,
        )
    )

    log.info(
        "graph.route",
        session=state.get("session_id"),
        direction=decision.direction.value,
        next_concept=decision.next_concept_key,
        reason=decision.reason[:120],
    )

    return {
        "last_decision": decision.model_dump(mode="json"),
        "interviewer_notes": [f"Route -> {decision.direction.value}: {decision.reason}"],
    }


def enough_evidence(state: InterviewState) -> str:
    """Conditional edge: keep interviewing, or write the report.

    Any one of these ends the interview:
      * the planned question budget is spent,
      * the clock has run out,
      * a hard cap on questions is hit (a runaway loop must never bill a candidate
        for a hundred questions).
    """
    settings = get_settings()
    asked = questions_asked(state)

    if asked >= settings.max_questions_per_interview:
        log.info("graph.stop", reason="max_questions", asked=asked)
        return "report"

    if questions_remaining(state) <= 0:
        log.info("graph.stop", reason="plan_complete", asked=asked)
        return "report"

    if minutes_remaining(state) <= 0.5 and asked >= settings.min_questions_per_interview:
        log.info("graph.stop", reason="time_exhausted", asked=asked)
        return "report"

    return "select_question"
