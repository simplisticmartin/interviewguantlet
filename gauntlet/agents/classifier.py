"""Response classifier.

Runs before evaluation so the graph can route: a clarifying question deserves an
answer, not a score; an honest "I don't know" is a real signal and must not be graded
as an off-topic ramble.
"""

from __future__ import annotations

import structlog

from gauntlet.agents.base import Agent
from gauntlet.llm.base import StructuredOutputError
from gauntlet.prompts.catalog import RESPONSE_CLASSIFIER
from gauntlet.schemas import AnswerPayload, QuestionSpec, ResponseClass, ResponseClassification

log = structlog.get_logger(__name__)


class ResponseClassifierAgent(Agent):
    key = "classifier"

    def classify(self, question: QuestionSpec, answer: AnswerPayload) -> ResponseClassification:
        if not answer.text.strip() and not answer.code:
            return ResponseClassification(response_class=ResponseClass.EMPTY)

        try:
            result = self.invoke(
                RESPONSE_CLASSIFIER,
                ResponseClassification,
                context={
                    "question": question.prompt_text,
                    "expects_code": question.expects_code,
                    "interview_type": question.interview_type.value,
                },
                blocks={"candidate_answer": answer.text},
            )
        except StructuredOutputError:
            log.warning("classifier.failed", question=question.prompt_text[:80])
            return ResponseClassification(response_class=ResponseClass.SUBSTANTIVE)

        classification = result.value
        # A submitted code block is ground truth about the response shape, whatever the
        # model decided from the prose.
        if answer.code and not classification.contains_code:
            classification = classification.model_copy(
                update={
                    "contains_code": True,
                    "detected_language": answer.language or classification.detected_language,
                }
            )
        return classification
