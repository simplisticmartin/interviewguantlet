"""Rubric-based evaluation: structured judgements, not holistic guesses."""

from gauntlet.evaluation.engine import JUDGES, EvaluationEngine, JudgeSpec, aggregate
from gauntlet.evaluation.rubrics import (
    RUBRICS,
    generic_rubric,
    get_rubric,
    rubric_for_concept,
    rubric_index,
)

__all__ = [
    "JUDGES",
    "RUBRICS",
    "EvaluationEngine",
    "JudgeSpec",
    "aggregate",
    "generic_rubric",
    "get_rubric",
    "rubric_for_concept",
    "rubric_index",
]
