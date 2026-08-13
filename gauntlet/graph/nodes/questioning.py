"""Question selection, asking, waiting for the candidate, and response routing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from langgraph.types import interrupt

from gauntlet.agents.classifier import ResponseClassifierAgent
from gauntlet.agents.interviewer import InterviewerAgent, QuestionTarget
from gauntlet.agents.personas import get_persona, persona_for_concept
from gauntlet.content.taxonomy import display_name
from gauntlet.evaluation.rubrics import rubric_for_concept
from gauntlet.graph.state import (
    InterviewState,
    asked_prompts,
    difficulty_label,
    elapsed_seconds,
    mode_of,
    remaining_seconds,
    skill_graph_from_state,
)
from gauntlet.prompts.catalog import CLARIFICATION_REPLY
from gauntlet.schemas import (
    AdaptiveDirection,
    AnswerPayload,
    ClarificationReply,
    InterviewType,
    QuestionSpec,
    ResponseClass,
)

log = structlog.get_logger(__name__)


def select_question(state: InterviewState) -> dict[str, Any]:
    """Choose what the next question examines.

    The adaptive decision from the previous turn wins; the plan's slate is the fallback
    when there is no reason to deviate.
    """
    plan = state.get("interview_plan") or {}
    slate: list[dict[str, Any]] = list(plan.get("slate", []))
    cursor = int(state.get("plan_cursor", 0))
    difficulty = int(state.get("difficulty", 3))
    decision = state.get("last_decision") or {}
    direction = decision.get("direction")
    graph = skill_graph_from_state(state)

    followups = int(state.get("followups_on_concept", 0))
    current_concept = state.get("current_concept_key")

    # --- Follow-up probe: stay on the concept that needs probing ---------
    if direction == AdaptiveDirection.PROBE.value and current_concept:
        # A misconception can surface about a *neighbouring* concept (asked about offset
        # commits, volunteered something wrong about ordering). Probe where the wrong
        # belief actually is, not where the question happened to be aimed.
        misconception = (state.get("last_evaluation") or {}).get("misconception") or {}
        probe_concept = current_concept
        if misconception.get("detected") and misconception.get("concept_key"):
            probe_concept = str(misconception["concept_key"])

        target = _target_from_concept(
            probe_concept,
            difficulty=difficulty,
            interview_type=(
                _type_of_current(state) if probe_concept == current_concept else None
            ),
            is_followup=True,
        )
        return {
            "pending_target": {**target, "is_followup": True},
            "followups_on_concept": followups + 1,
            "current_concept_key": probe_concept,
        }

    # --- Descend into a deeper sub-concept -------------------------------
    if direction == AdaptiveDirection.DEEPER.value and decision.get("next_concept_key"):
        concept_key = str(decision["next_concept_key"])
        new_difficulty = _clamp_difficulty(difficulty + int(decision.get("difficulty_delta", 1)))
        target = _target_from_concept(concept_key, difficulty=new_difficulty)
        return {
            "pending_target": target,
            "difficulty": new_difficulty,
            "followups_on_concept": 0,
            "current_concept_key": concept_key,
        }

    # --- Otherwise walk the slate ----------------------------------------
    delta = int(decision.get("difficulty_delta", 0)) if decision else 0
    new_difficulty = _clamp_difficulty(difficulty + delta)

    slot: dict[str, Any] | None = None
    while cursor < len(slate):
        candidate = slate[cursor]
        cursor += 1
        keys = list(candidate.get("concept_keys", []))
        # Don't spend a slot on something already well evidenced.
        if keys and graph.is_saturated(keys[0]):
            log.info("graph.slate.skip_saturated", concept=keys[0])
            continue
        slot = candidate
        break

    if slot is None:
        # Slate exhausted but the interview still has budget: revisit the weakest
        # observed concept at a lower difficulty to find the floor.
        weakest = graph.weakest(limit=1)
        concept_key = weakest[0].concept_key if weakest else ""
        target = _target_from_concept(
            concept_key or "system_design.api", difficulty=_clamp_difficulty(new_difficulty - 1)
        )
        return {
            "pending_target": target,
            "plan_cursor": cursor,
            "difficulty": target["difficulty"],
            "followups_on_concept": 0,
            "current_concept_key": target["concept_keys"][0] if target["concept_keys"] else None,
        }

    keys = list(slot.get("concept_keys", []))
    # The router's difficulty is authoritative and carries across slots; the slot only
    # clamps it into the band the concept can actually be asked at. Averaging the two
    # here would silently cancel the router's decision on every turn.
    resolved_difficulty = _clamp_to_concept(keys[0] if keys else "", new_difficulty)

    target = {
        "concept_keys": keys,
        "interview_type": slot.get("interview_type", InterviewType.JAVA.value),
        "difficulty": resolved_difficulty,
        "rubric_key": slot.get("rubric_key"),
        "is_resume_probe": bool(slot.get("is_resume_probe", False)),
        "claim_text": slot.get("claim_text"),
        "claim_has_metric": bool(slot.get("claim_has_metric", False)),
        "ask_confidence": bool(slot.get("ask_confidence", False)),
        "is_followup": False,
    }
    return {
        "pending_target": target,
        "plan_cursor": cursor,
        "difficulty": resolved_difficulty,
        "followups_on_concept": 0,
        "current_concept_key": keys[0] if keys else None,
    }


def ask_question(state: InterviewState) -> dict[str, Any]:
    """Author the question and put it to the candidate."""
    target_dict = state.get("pending_target") or {}
    target = QuestionTarget(
        concept_keys=list(target_dict.get("concept_keys", [])),
        interview_type=_interview_type(target_dict.get("interview_type")),
        difficulty=int(target_dict.get("difficulty", 3)),
        rubric_key=target_dict.get("rubric_key"),
        is_resume_probe=bool(target_dict.get("is_resume_probe", False)),
        claim_text=target_dict.get("claim_text"),
        claim_has_metric=bool(target_dict.get("claim_has_metric", False)),
        ask_confidence=bool(target_dict.get("ask_confidence", False)),
    )

    interviewer = InterviewerAgent(mode=mode_of(state))
    history = asked_prompts(state)

    if target_dict.get("is_followup") and state.get("current_question"):
        from gauntlet.schemas import AggregateEvaluation

        previous = state["current_question"] or {}
        evaluation = AggregateEvaluation.model_validate(
            state.get("last_evaluation") or {"score": 0.0}
        )
        last_answer = (state.get("answer_history") or [{}])[-1]
        spec = interviewer.probe(
            target,
            evaluation,
            question_text=str(previous.get("prompt_text", "")),
            answer_text=str(last_answer.get("text", "")),
            asked_prompts=history,
            probe_reason=str((state.get("last_decision") or {}).get("reason", "")),
        )
    else:
        spec = interviewer.ask(
            target,
            asked_prompts=history,
            candidate_context=_candidate_context(state, target),
            target_level=state.get("target_level", "senior"),
            target_role=state.get("target_role", "Software Engineer"),
        )

    ordinal = len(state.get("question_history", [])) + 1
    record = {
        **spec.model_dump(mode="json"),
        "ordinal": ordinal,
        "asked_at": datetime.now(UTC).isoformat(),
        "difficulty_label": difficulty_label(spec.difficulty),
        "claim_text": target.claim_text,
        "is_resume_probe": target.is_resume_probe,
    }

    log.info(
        "graph.ask_question",
        session=state.get("session_id"),
        ordinal=ordinal,
        concept=target.primary_concept,
        difficulty=spec.difficulty,
        followup=spec.is_followup,
        agent=spec.agent_key,
    )

    return {
        "current_question": record,
        "question_history": [record],
        "pending_answer": None,
        "pending_clarification": None,
        "elapsed_time": elapsed_seconds(state),
        "remaining_time": remaining_seconds(state),
        "status": "awaiting_answer",
    }


def wait_for_candidate(state: InterviewState) -> dict[str, Any]:
    """Suspend the graph until an answer arrives.

    ``interrupt`` checkpoints here, so the process can restart, the candidate can close
    the tab and come back, and the interview resumes exactly where it stopped.
    """
    question = state.get("current_question") or {}
    payload = interrupt(
        {
            "type": "question",
            "ordinal": question.get("ordinal"),
            "prompt_text": question.get("prompt_text"),
            "interview_type": question.get("interview_type"),
            "agent_key": question.get("agent_key"),
            "expects_code": question.get("expects_code", False),
            "asks_confidence": question.get("asks_confidence", False),
            "remaining_seconds": remaining_seconds(state),
        }
    )
    answer = payload if isinstance(payload, dict) else {"text": str(payload)}
    return {"pending_answer": answer, "status": "in_progress"}


def classify_response(state: InterviewState) -> dict[str, Any]:
    """Decide what kind of response arrived before anything is graded."""
    question = QuestionSpec.model_validate(_question_spec_dict(state))
    answer = AnswerPayload.model_validate(state.get("pending_answer") or {})

    classification = ResponseClassifierAgent().classify(question, answer)
    log.info(
        "graph.classify",
        session=state.get("session_id"),
        response_class=classification.response_class.value,
    )
    return {"last_classification": classification.model_dump(mode="json")}


def answer_clarification(state: InterviewState) -> dict[str, Any]:
    """Answer a clarifying question, then hand the floor back to the candidate."""
    question = _question_spec_dict(state)
    answer = AnswerPayload.model_validate(state.get("pending_answer") or {})
    concept_keys = list(question.get("concept_keys", []))
    persona = (
        persona_for_concept(concept_keys[0], _interview_type(question.get("interview_type")))
        if concept_keys
        else get_persona(str(question.get("agent_key", "java")))
    )

    agent = ResponseClassifierAgent()
    result = agent.invoke(
        CLARIFICATION_REPLY,
        ClarificationReply,
        system_vars={"persona": persona.system},
        context={
            "question": question.get("prompt_text", ""),
            "interview_type": question.get("interview_type"),
            "concept_keys": concept_keys,
            "mode": state.get("mode", "real"),
        },
        blocks={"candidate_answer": answer.text},
    )
    reply = result.value

    log.info("graph.clarification", session=state.get("session_id"))
    return {
        "pending_clarification": {
            **reply.model_dump(mode="json"),
            "at": datetime.now(UTC).isoformat(),
        },
        "pending_answer": None,
        # A clarification that leaked part of the answer counts as a hint against the
        # eventual evidence weight.
        "interviewer_notes": (
            ["Clarification given that partially revealed the expected answer."]
            if reply.gave_away_answer
            else []
        ),
        "status": "awaiting_answer",
    }


def wait_after_clarification(state: InterviewState) -> dict[str, Any]:
    """Second interrupt point: the candidate answers after being clarified."""
    clarification = state.get("pending_clarification") or {}
    question = state.get("current_question") or {}
    payload = interrupt(
        {
            "type": "clarification",
            "ordinal": question.get("ordinal"),
            "reply": clarification.get("reply", ""),
            "prompt_text": question.get("prompt_text"),
            "remaining_seconds": remaining_seconds(state),
        }
    )
    answer = payload if isinstance(payload, dict) else {"text": str(payload)}
    return {"pending_answer": answer, "status": "in_progress"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def route_after_classification(state: InterviewState) -> str:
    """Conditional edge: clarifying questions get answered, everything else graded."""
    classification = state.get("last_classification") or {}
    if classification.get("response_class") == ResponseClass.CLARIFYING_QUESTION.value:
        return "answer_clarification"
    if classification.get("contains_code"):
        return "check_code"
    return "evaluate_answer"


def _clamp_difficulty(value: int) -> int:
    return max(1, min(5, value))


def _clamp_to_concept(concept_key: str, difficulty: int) -> int:
    """Keep difficulty inside the band the concept supports.

    You cannot ask a difficulty-1 question about the Java memory model, and a
    difficulty-5 question about Optional does not exist.
    """
    from gauntlet.content.taxonomy import get_concept

    value = _clamp_difficulty(difficulty)
    concept = get_concept(concept_key)
    if concept is None:
        return value
    return max(concept.difficulty_floor, min(concept.difficulty_ceiling, value))


def _interview_type(value: Any) -> InterviewType:
    try:
        return InterviewType(str(value))
    except ValueError:
        return InterviewType.JAVA


def _type_of_current(state: InterviewState) -> InterviewType:
    question = state.get("current_question") or {}
    return _interview_type(question.get("interview_type"))


def _target_from_concept(
    concept_key: str,
    difficulty: int,
    interview_type: InterviewType | None = None,
    is_followup: bool = False,
) -> dict[str, Any]:
    from gauntlet.content.taxonomy import get_concept, is_branch
    from gauntlet.graph.slate import resolve_examinable

    # The router may hand back a branch concept; resolve it to something askable.
    if concept_key and is_branch(concept_key) and not is_followup:
        candidates = resolve_examinable([concept_key], difficulty)
        concept_key = candidates[0] if candidates else concept_key

    concept = get_concept(concept_key)
    resolved_type = interview_type or (concept.interview_type if concept else InterviewType.JAVA)
    rubric = rubric_for_concept(concept_key, resolved_type)
    return {
        "concept_keys": [concept_key] if concept_key else [],
        "interview_type": resolved_type.value,
        "difficulty": _clamp_to_concept(concept_key, difficulty),
        "rubric_key": rubric.key,
        "is_resume_probe": False,
        "claim_text": None,
        "ask_confidence": False,
        "is_followup": is_followup,
    }


def _question_spec_dict(state: InterviewState) -> dict[str, Any]:
    """The current question as a QuestionSpec-shaped dict (history fields removed)."""
    question = dict(state.get("current_question") or {})
    for key in ("ordinal", "asked_at", "difficulty_label", "claim_text", "is_resume_probe"):
        question.pop(key, None)
    question.setdefault("prompt_text", "(question unavailable)")
    question.setdefault("interview_type", InterviewType.JAVA.value)
    question.setdefault("agent_key", "java")
    return question


def _candidate_context(state: InterviewState, target: QuestionTarget) -> str:
    """Untrusted candidate material relevant to this question, for grounding.

    Only what the question actually needs: the resume claim under examination, or a
    short profile summary. Never the whole resume on every turn.
    """
    if target.is_resume_probe and target.claim_text:
        return f"Resume claim under discussion: {target.claim_text}"

    profile = state.get("resume_profile") or {}
    bits: list[str] = []
    if profile.get("headline"):
        bits.append(str(profile["headline"]))
    if profile.get("years_experience"):
        bits.append(f"{profile['years_experience']} years experience")
    relevant = [
        claim.get("claim_text", "")
        for claim in profile.get("claims", [])
        if set(claim.get("concept_keys", [])) & set(target.concept_keys)
    ]
    bits.extend(relevant[:2])
    if target.primary_concept:
        bits.append(f"Concept under examination: {display_name(target.primary_concept)}")
    return "\n".join(bit for bit in bits if bit)
