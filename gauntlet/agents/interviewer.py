"""The interviewer: authors the next question and the adversarial follow-ups.

Which specialist voice speaks is decided by the concept under examination, so the
Java persona asks the ConcurrentHashMap question and the distributed-systems persona
asks the Kafka one - each probing for what that specialism actually cares about
(spec sections 4 and 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from gauntlet.agents.base import Agent
from gauntlet.agents.personas import Persona, persona_for_concept, persona_for_type
from gauntlet.content.taxonomy import display_name
from gauntlet.evaluation.rubrics import get_rubric
from gauntlet.llm.base import LLMProvider
from gauntlet.prompts.catalog import FOLLOWUP_PROBE, QUESTION_AUTHOR
from gauntlet.retrieval.question_index import get_question_index
from gauntlet.schemas import (
    AggregateEvaluation,
    InterviewMode,
    InterviewType,
    QuestionSpec,
    RubricSpec,
)


@dataclass(frozen=True, slots=True)
class QuestionTarget:
    """What the next question should examine."""

    concept_keys: list[str]
    interview_type: InterviewType
    difficulty: int
    rubric_key: str | None = None
    is_resume_probe: bool = False
    claim_text: str | None = None
    claim_has_metric: bool = False
    ask_confidence: bool = False

    @property
    def primary_concept(self) -> str:
        return self.concept_keys[0] if self.concept_keys else ""


class InterviewerAgent(Agent):
    """Authors questions.

    Holds only the set of things already asked, so it can avoid repeating itself
    within a session. All other context arrives per call from the graph state.
    """

    key = "interviewer"

    def __init__(
        self,
        mode: InterviewMode = InterviewMode.REAL,
        provider: LLMProvider | None = None,
    ) -> None:
        super().__init__(provider)
        self.mode = mode
        self._asked: set[str] = set()

    # --- Opening / next question ----------------------------------------

    def ask(
        self,
        target: QuestionTarget,
        *,
        asked_prompts: list[str],
        candidate_context: str = "",
        target_level: str = "senior",
        target_role: str = "Software Engineer",
    ) -> QuestionSpec:
        persona = self._persona(target)
        rubric = get_rubric(target.rubric_key, target.interview_type)

        retrieved = get_question_index().for_concepts(
            concept_keys=target.concept_keys,
            difficulty=target.difficulty,
            interview_type=target.interview_type,
            exclude_slugs=frozenset(self._asked),
            limit=6,
        )

        result = self.invoke(
            QUESTION_AUTHOR,
            QuestionSpec,
            system_vars={"persona": persona.system},
            context={
                "target": {
                    "concept_key": target.primary_concept,
                    "display_name": display_name(target.primary_concept),
                    "concept_keys": target.concept_keys,
                    "interview_type": target.interview_type.value,
                    "agent_key": persona.key,
                    "difficulty": target.difficulty,
                    "rubric_key": rubric.key,
                },
                "target_role": target_role,
                "target_level": target_level,
                "mode": self.mode.value,
                "is_resume_probe": target.is_resume_probe,
                "resume_claim": target.claim_text,
                "claim_has_metric": target.claim_has_metric,
                "asks_confidence": target.ask_confidence,
                "rubric_dimensions": [
                    {"key": d.key, "label": d.label, "probe": d.hint} for d in rubric.dimensions
                ],
                "asked_prompts": asked_prompts,
                "candidate_questions": [item.as_context() for item in retrieved],
                "corpus_note": (
                    "candidate_questions are Gauntlet-authored reference questions, not "
                    "questions attributed to any company. Adapt or replace them."
                ),
            },
            blocks={"candidate_context": candidate_context} if candidate_context else None,
        )

        spec = self._finalise(result.value, target, persona, rubric)
        self._asked.add(spec.prompt_text)
        for item in retrieved:
            if item.seed.question == spec.prompt_text:
                self._asked.add(item.seed.slug)
        return spec

    # --- Adversarial follow-up ------------------------------------------

    def probe(
        self,
        target: QuestionTarget,
        evaluation: AggregateEvaluation,
        *,
        question_text: str,
        answer_text: str,
        asked_prompts: list[str],
        probe_reason: str = "",
    ) -> QuestionSpec:
        """Ask the question whose answer separates real understanding from recall."""
        persona = self._persona(target)
        rubric = get_rubric(target.rubric_key, target.interview_type)

        gaps = set(evaluation.missing) | set(evaluation.incorrect)
        # Each rubric dimension carries an authored probe question in `hint`; using them
        # keeps offline follow-ups sharp and gives the model concrete angles to pick from.
        gap_probes = [
            dimension.hint
            for dimension in rubric.dimensions
            if dimension.key in gaps and dimension.hint
        ]
        if evaluation.misconception.detected:
            gap_probes.insert(
                0,
                _misconception_probe(evaluation, rubric),
            )

        result = self.invoke(
            FOLLOWUP_PROBE,
            QuestionSpec,
            system_vars={"persona": persona.system},
            context={
                "target": {
                    "concept_key": target.primary_concept,
                    "display_name": display_name(target.primary_concept),
                    "concept_keys": target.concept_keys,
                    "interview_type": target.interview_type.value,
                    "agent_key": persona.key,
                    "difficulty": target.difficulty,
                    "rubric_key": rubric.key,
                },
                "original_question": question_text,
                "score": round(evaluation.score, 3),
                "missing_dimensions": [
                    {"key": d.key, "label": d.label}
                    for d in rubric.dimensions
                    if d.key in evaluation.missing
                ],
                "incorrect_dimensions": evaluation.incorrect,
                "misconception": evaluation.misconception.model_dump()
                if evaluation.misconception.detected
                else None,
                "gap_probes": gap_probes,
                "probe_reason": probe_reason or "verify depth behind the answer",
                "asked_prompts": asked_prompts,
            },
            blocks={"candidate_answer": answer_text},
        )

        spec = self._finalise(result.value, target, persona, rubric)
        spec = spec.model_copy(
            update={
                "is_followup": True,
                "probe_reason": probe_reason or spec.probe_reason or "depth probe",
                "asks_confidence": False,
            }
        )
        self._asked.add(spec.prompt_text)
        return spec

    # --- Helpers ---------------------------------------------------------

    def _persona(self, target: QuestionTarget) -> Persona:
        if target.is_resume_probe:
            from gauntlet.agents.personas import get_persona

            return get_persona("resume_defense")
        if target.primary_concept:
            return persona_for_concept(target.primary_concept, target.interview_type)
        return persona_for_type(target.interview_type)

    def _finalise(
        self,
        spec: QuestionSpec,
        target: QuestionTarget,
        persona: Persona,
        rubric: RubricSpec,
    ) -> QuestionSpec:
        """Pin the fields the system owns; the model only owns the wording."""
        return spec.model_copy(
            update={
                "interview_type": target.interview_type,
                "agent_key": persona.key,
                "concept_keys": target.concept_keys or spec.concept_keys,
                "difficulty": target.difficulty,
                "rubric_key": rubric.key,
                "asks_confidence": target.ask_confidence or spec.asks_confidence,
            }
        )


def _misconception_probe(evaluation: AggregateEvaluation, rubric: RubricSpec) -> str:
    """Turn a detected false belief into a scenario that exposes it.

    Deliberately does not correct the candidate - it hands them a situation where their
    stated belief produces a visibly wrong outcome and lets them work it out.
    """
    belief = evaluation.misconception.belief.strip().rstrip(".")
    for dimension in rubric.dimensions:
        if dimension.key in evaluation.incorrect and dimension.hint:
            return dimension.hint
    if belief:
        return (
            f"You said: {belief}. Walk me through what a client would actually observe "
            "if that were true and the system were under load."
        )
    return "Walk me through a concrete case where that assumption gets tested."
