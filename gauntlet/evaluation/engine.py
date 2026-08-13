"""Structured evaluation engine (spec sections 18-19).

Two things make this different from "rate this answer 1-10":

1. **Rubrics.** Grading is a set of named judgements about named dimensions, so a score
   can always be decomposed into what was shown, what was missed, and what was wrong.
2. **Independent judges.** Technical accuracy, reasoning, communication, and the hiring
   bar are graded separately and then aggregated. Their *disagreement* is retained as a
   signal: when judges diverge, the aggregate confidence drops, which is exactly when a
   human should not trust the number.

Judges never see each other's verdicts, and the interviewer persona is deliberately kept
out of grading so interviewer style cannot contaminate the score.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import structlog

from gauntlet.agents.base import Agent
from gauntlet.config import get_settings
from gauntlet.evaluation.rubrics import get_rubric, misconception_candidates
from gauntlet.llm.base import LLMProvider, StructuredOutputError
from gauntlet.prompts.catalog import (
    JUDGE_COMMUNICATION,
    JUDGE_HIRING_BAR,
    JUDGE_REASONING,
    JUDGE_TECHNICAL,
    MISCONCEPTION_DETECTOR,
)
from gauntlet.prompts.registry import PromptTemplate
from gauntlet.schemas import (
    AggregateEvaluation,
    AnswerPayload,
    JudgeVerdict,
    MisconceptionFinding,
    QuestionSpec,
    RubricSpec,
)

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    key: str
    template: PromptTemplate
    weight: float
    scores_communication: bool = False


# Weights for aggregating the score. Communication is reported separately rather than
# diluting technical correctness - a beautifully explained wrong answer is still wrong.
JUDGES: tuple[JudgeSpec, ...] = (
    JudgeSpec("technical_accuracy", JUDGE_TECHNICAL, 0.45),
    JudgeSpec("reasoning", JUDGE_REASONING, 0.30),
    JudgeSpec("hiring_bar", JUDGE_HIRING_BAR, 0.25),
    JudgeSpec("communication", JUDGE_COMMUNICATION, 0.0, scores_communication=True),
)

PRIMARY_JUDGE = "technical_accuracy"


class EvaluationEngine(Agent):
    key = "evaluator"

    def __init__(
        self, provider: LLMProvider | None = None, multi_judge: bool | None = None
    ) -> None:
        super().__init__(provider)
        self._multi_judge = (
            get_settings().multi_judge_enabled if multi_judge is None else multi_judge
        )

    def evaluate(
        self,
        question: QuestionSpec,
        answer: AnswerPayload,
        *,
        target_role: str = "Software Engineer",
        target_level: str = "senior",
    ) -> AggregateEvaluation:
        rubric = get_rubric(question.rubric_key, question.interview_type)
        panel = JUDGES if self._multi_judge else JUDGES[:1]

        verdicts: list[JudgeVerdict] = []
        for judge in panel:
            verdict = self._run_judge(judge, question, answer, rubric, target_role, target_level)
            if verdict is not None:
                verdicts.append(verdict)

        if not verdicts:
            # Every judge failed. Return an explicitly low-confidence zero rather than
            # inventing a score - downstream weighting will then largely ignore it.
            return AggregateEvaluation(
                score=0.0,
                confidence=0.0,
                missing=rubric.dimension_keys(),
                verdicts=[],
            )

        misconception = self._detect_misconception(question, answer, rubric)
        return aggregate(verdicts, misconception)

    # --- Internals -------------------------------------------------------

    def _run_judge(
        self,
        judge: JudgeSpec,
        question: QuestionSpec,
        answer: AnswerPayload,
        rubric: RubricSpec,
        target_role: str,
        target_level: str,
    ) -> JudgeVerdict | None:
        try:
            result = self.invoke(
                judge.template,
                JudgeVerdict,
                context={
                    "judge_key": judge.key,
                    "question": question.prompt_text,
                    "interview_type": question.interview_type.value,
                    "concept_keys": question.concept_keys,
                    "difficulty": question.difficulty,
                    "target_role": target_role,
                    "target_level": target_level,
                    "self_reported_confidence": answer.self_confidence,
                    "rubric": rubric.model_dump(),
                },
                blocks={"candidate_answer": _answer_blob(answer)},
            )
        except StructuredOutputError:
            # One judge failing must not fail the interview; the panel degrades instead.
            log.warning(
                "evaluation.judge.failed",
                judge=judge.key,
                question=question.prompt_text[:80],
            )
            return None

        verdict = result.value
        return verdict.model_copy(update={"judge_key": judge.key})

    def _detect_misconception(
        self, question: QuestionSpec, answer: AnswerPayload, rubric: RubricSpec
    ) -> MisconceptionFinding:
        if not answer.text.strip():
            return MisconceptionFinding(detected=False)

        # Widen beyond this question's rubric: candidates volunteer wrong beliefs about
        # neighbouring topics, and those are worth catching wherever they surface.
        patterns = misconception_candidates(question.concept_keys, rubric)
        rubric_payload = rubric.model_dump()
        rubric_payload["common_misconceptions"] = [
            pattern.model_dump() for pattern in patterns
        ]

        try:
            result = self.invoke(
                MISCONCEPTION_DETECTOR,
                MisconceptionFinding,
                context={
                    "question": question.prompt_text,
                    "concept_key": question.concept_keys[0] if question.concept_keys else None,
                    "concept_keys": question.concept_keys,
                    "self_reported_confidence": answer.self_confidence,
                    "rubric": rubric_payload,
                },
                blocks={"candidate_answer": _answer_blob(answer)},
            )
        except StructuredOutputError:
            log.warning("evaluation.misconception.failed", question=question.prompt_text[:80])
            return MisconceptionFinding(detected=False)

        finding = result.value
        if finding.detected and not finding.concept_key and question.concept_keys:
            finding = finding.model_copy(update={"concept_key": question.concept_keys[0]})
        return finding


def _answer_blob(answer: AnswerPayload) -> str:
    parts = [answer.text.strip()]
    if answer.code:
        language = answer.language or ""
        parts.append(f"\n\nSubmitted code ({language}):\n```{language}\n{answer.code}\n```")
    return "".join(part for part in parts if part)


def aggregate(
    verdicts: list[JudgeVerdict],
    misconception: MisconceptionFinding | None = None,
) -> AggregateEvaluation:
    """Fuse judge verdicts into the single reading the skill graph consumes."""
    by_key = {verdict.judge_key: verdict for verdict in verdicts}
    weights = {judge.key: judge.weight for judge in JUDGES}

    scoring = [
        (verdict, weights.get(verdict.judge_key, 0.0))
        for verdict in verdicts
        if weights.get(verdict.judge_key, 0.0) > 0
    ]
    if scoring:
        total_weight = sum(weight for _, weight in scoring)
        score = sum(verdict.score * weight for verdict, weight in scoring) / total_weight
    else:
        score = statistics.fmean(verdict.score for verdict in verdicts)

    scores = [verdict.score for verdict, _ in scoring] or [v.score for v in verdicts]
    disagreement = statistics.pstdev(scores) if len(scores) > 1 else 0.0

    mean_confidence = statistics.fmean(verdict.confidence for verdict in verdicts)
    # Judges pulling in different directions is itself evidence the score is soft.
    confidence = max(0.0, min(1.0, mean_confidence * (1.0 - min(disagreement * 1.5, 0.6))))

    primary = by_key.get(PRIMARY_JUDGE) or verdicts[0]
    demonstrated = list(dict.fromkeys(primary.demonstrated))
    incorrect = list(
        dict.fromkeys(item for verdict in verdicts for item in verdict.incorrect)
    )
    # Missing only counts if no judge saw it demonstrated.
    seen_anywhere = {item for verdict in verdicts for item in verdict.demonstrated}
    missing = [item for item in primary.missing if item not in seen_anywhere]

    communication = next(
        (
            verdict.communication_score
            for verdict in verdicts
            if verdict.judge_key == "communication" and verdict.communication_score is not None
        ),
        None,
    )
    if communication is None:
        scores_present = [
            verdict.communication_score
            for verdict in verdicts
            if verdict.communication_score is not None
        ]
        communication = statistics.fmean(scores_present) if scores_present else None

    finding = misconception or MisconceptionFinding(detected=False)
    if finding.detected:
        # A confidently-wrong answer must not be rescued by a strong communication or
        # reasoning judge (spec section 22: this is the quadrant that matters most).
        score = min(score, 0.45)

    return AggregateEvaluation(
        score=max(0.0, min(1.0, score)),
        communication_score=communication,
        confidence=confidence,
        demonstrated=demonstrated,
        missing=missing,
        incorrect=incorrect,
        verdicts=verdicts,
        disagreement=round(disagreement, 4),
        misconception=finding,
    )
