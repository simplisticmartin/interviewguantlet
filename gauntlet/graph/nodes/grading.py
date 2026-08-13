"""Grading nodes: static code check, evaluation, skill-graph update, misconception log."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from gauntlet.evaluation.engine import EvaluationEngine
from gauntlet.execution.static_check import check_code
from gauntlet.graph.nodes.questioning import _question_spec_dict
from gauntlet.graph.state import (
    InterviewState,
    elapsed_seconds,
    remaining_seconds,
    skill_graph_from_state,
)
from gauntlet.schemas import AnswerPayload, QuestionSpec, ResponseClass
from gauntlet.skills.graph import SkillGraph

log = structlog.get_logger(__name__)


def check_submitted_code(state: InterviewState) -> dict[str, Any]:
    """Statically analyse submitted code.

    Nothing is executed here (see :mod:`gauntlet.execution.static_check`). The signals
    this produces - unparseable, nested loops, no empty-input guard - are what the
    interviewer uses to choose its next question, which is exactly how a human
    interviewer reads a submission before the tests have even run.
    """
    answer = AnswerPayload.model_validate(state.get("pending_answer") or {})
    source = answer.code or answer.text
    if not source.strip():
        return {"code_check": None}

    result = check_code(source, answer.language)
    log.info(
        "graph.code_check",
        session=state.get("session_id"),
        language=result.language,
        syntax_ok=result.syntax_ok,
        loop_depth=result.max_loop_depth,
    )
    return {
        "code_check": result.as_dict(),
        "interviewer_notes": [f"Code check: {signal}" for signal in result.interviewer_signals],
    }


def evaluate_answer(state: InterviewState) -> dict[str, Any]:
    """Run the judge panel over the answer."""
    question = QuestionSpec.model_validate(_question_spec_dict(state))
    answer = AnswerPayload.model_validate(state.get("pending_answer") or {})
    classification = state.get("last_classification") or {}

    evaluation = EvaluationEngine().evaluate(
        question,
        answer,
        target_role=state.get("target_role", "Software Engineer"),
        target_level=state.get("target_level", "senior"),
    )

    # An explicit "I don't know" is honest and must not be scored as a wrong answer -
    # but it is still zero demonstrated knowledge.
    if classification.get("response_class") == ResponseClass.DONT_KNOW.value:
        evaluation = evaluation.model_copy(
            update={
                "score": 0.0,
                "incorrect": [],
                "confidence": max(evaluation.confidence, 0.8),
                "misconception": evaluation.misconception.model_copy(
                    update={"detected": False}
                ),
            }
        )

    code_check = state.get("code_check") or {}
    if code_check and not code_check.get("syntax_ok", True):
        evaluation = evaluation.model_copy(
            update={"score": min(evaluation.score, 0.5)}
        )

    ordinal = len(state.get("question_history", []))
    answer_record = {
        **answer.model_dump(mode="json"),
        "ordinal": ordinal,
        "answered_at": datetime.now(UTC).isoformat(),
        "score": round(evaluation.score, 3),
        "response_class": classification.get("response_class"),
    }

    log.info(
        "graph.evaluate",
        session=state.get("session_id"),
        ordinal=ordinal,
        score=round(evaluation.score, 3),
        confidence=round(evaluation.confidence, 3),
        disagreement=evaluation.disagreement,
        judges=len(evaluation.verdicts),
    )

    return {
        "last_evaluation": evaluation.model_dump(mode="json"),
        "answer_history": [answer_record],
        "elapsed_time": elapsed_seconds(state),
        "remaining_time": remaining_seconds(state),
    }


def update_skill_graph(state: InterviewState) -> dict[str, Any]:
    """Record the evidence and recompute the live skill picture."""
    from gauntlet.schemas import AggregateEvaluation

    evaluation = AggregateEvaluation.model_validate(state.get("last_evaluation") or {"score": 0.0})
    question = state.get("current_question") or {}
    answer = state.get("pending_answer") or {}
    concept_keys = list(question.get("concept_keys", []))

    if not concept_keys:
        return {"interviewer_notes": ["Question had no concept mapping; evidence not recorded."]}

    hints = 1 if (state.get("pending_clarification") or {}).get("gave_away_answer") else 0

    evidence_row = {
        "concept_keys": concept_keys,
        "score": round(evaluation.score, 4),
        "difficulty": int(question.get("difficulty", 3)),
        "observed_at": datetime.now(UTC).isoformat(),
        "self_confidence": answer.get("self_confidence"),
        "hints_used": hints,
        "is_followup": bool(question.get("is_followup", False)),
        "judge_confidence": round(evaluation.confidence, 4),
        "ordinal": question.get("ordinal"),
    }

    # Rebuild including the new row so the returned scores reflect this answer.
    graph: SkillGraph = skill_graph_from_state(state)
    _record_row(graph, evidence_row)

    readings = graph.all_readings()
    skill_scores = {reading.concept_key: round(reading.mastery, 4) for reading in readings}
    confidence_scores = {reading.concept_key: round(reading.confidence, 4) for reading in readings}

    return {
        "evidence": [evidence_row],
        "skill_scores": skill_scores,
        "confidence_scores": confidence_scores,
    }


def misconception_check(state: InterviewState) -> dict[str, Any]:
    """Persist any confidently-wrong belief the judges surfaced."""
    from gauntlet.schemas import AggregateEvaluation

    evaluation = AggregateEvaluation.model_validate(state.get("last_evaluation") or {"score": 0.0})
    finding = evaluation.misconception
    if not finding.detected:
        return {}

    question = state.get("current_question") or {}
    record = {
        **finding.model_dump(mode="json"),
        "ordinal": question.get("ordinal"),
        "question": question.get("prompt_text"),
        "detected_at": datetime.now(UTC).isoformat(),
    }

    log.info(
        "graph.misconception",
        session=state.get("session_id"),
        concept=finding.concept_key,
        severity=finding.severity,
    )

    return {
        "misconceptions": [record],
        "interviewer_notes": [
            f"Misconception ({finding.concept_key}): {finding.belief}"
        ],
    }


def _record_row(graph: SkillGraph, row: dict[str, Any]) -> None:
    from gauntlet.skills.mastery import Evidence

    observed_at = datetime.fromisoformat(row["observed_at"])
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    graph.record(
        list(row["concept_keys"]),
        Evidence(
            score=float(row["score"]),
            difficulty=int(row["difficulty"]),
            observed_at=observed_at,
            self_confidence=row.get("self_confidence"),
            hints_used=int(row.get("hints_used", 0)),
            is_followup=bool(row.get("is_followup", False)),
            judge_confidence=float(row.get("judge_confidence", 1.0)),
        ),
    )
