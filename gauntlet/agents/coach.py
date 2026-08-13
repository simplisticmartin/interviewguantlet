"""Coaching Mode teaching (spec section 25).

Only ever runs when the session mode is ``coaching``. Real Interview Mode must stay
silent: the moment you teach into an assessment you stop measuring the thing you were
assessing, and the candidate's later answers are contaminated by your feedback.

The separation is structural rather than a prompt instruction - the graph does not route
through this node at all unless coaching is enabled.
"""

from __future__ import annotations

import structlog

from gauntlet.agents.base import Agent
from gauntlet.agents.personas import get_persona, persona_for_concept
from gauntlet.evaluation.rubrics import get_rubric
from gauntlet.llm.base import StructuredOutputError
from gauntlet.prompts.catalog import COACHING_FEEDBACK
from gauntlet.schemas import AggregateEvaluation, CoachingNote, QuestionSpec

log = structlog.get_logger(__name__)


class CoachAgent(Agent):
    key = "coach"

    def coach(
        self,
        question: QuestionSpec,
        evaluation: AggregateEvaluation,
        answer_text: str,
    ) -> CoachingNote:
        rubric = get_rubric(question.rubric_key, question.interview_type)
        persona = (
            persona_for_concept(question.concept_keys[0], question.interview_type)
            if question.concept_keys
            else get_persona(question.agent_key)
        )

        try:
            result = self.invoke(
                COACHING_FEEDBACK,
                CoachingNote,
                system_vars={"persona": persona.system},
                context={
                    "question": question.prompt_text,
                    "concept_keys": question.concept_keys,
                    "rubric": rubric.model_dump(),
                    "last_evaluation": {
                        # The note must never quote a score back to the candidate; the
                        # prompt forbids it and the score is omitted here so it cannot.
                        "demonstrated": evaluation.demonstrated,
                        "missing": evaluation.missing,
                        "incorrect": evaluation.incorrect,
                        "misconception": evaluation.misconception.model_dump(),
                    },
                },
                blocks={"candidate_answer": answer_text},
            )
        except StructuredOutputError:
            log.warning("coach.failed", question=question.prompt_text[:80])
            return _fallback_note(evaluation)

        return result.value


def _fallback_note(evaluation: AggregateEvaluation) -> CoachingNote:
    """Still useful teaching when the model call fails - it has the rubric verdict."""
    if evaluation.misconception.detected:
        return CoachingNote(
            feedback=(
                f'Worth correcting before we move on: you said "'
                f'{evaluation.misconception.belief}" - {evaluation.misconception.correction}'
            ),
            key_correction=evaluation.misconception.correction,
        )
    if evaluation.missing:
        return CoachingNote(
            feedback=(
                "The main thing missing there: "
                + ", ".join(str(item).replace("_", " ") for item in evaluation.missing[:2])
            )
        )
    return CoachingNote(feedback="Solid answer - let's keep going.")
