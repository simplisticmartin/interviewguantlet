"""Rubric lookup, judge aggregation, and misconception scoping."""

from __future__ import annotations

import pytest

from gauntlet.evaluation.engine import aggregate
from gauntlet.evaluation.rubrics import (
    RUBRICS,
    generic_rubric,
    get_rubric,
    misconception_candidates,
    rubric_for_concept,
)
from gauntlet.schemas import InterviewType, JudgeVerdict, MisconceptionFinding


class TestRubricLookup:
    def test_exact_key_wins(self):
        rubric = get_rubric("kafka.ordering", InterviewType.DISTRIBUTED)
        assert rubric.key == "kafka.ordering"

    def test_unknown_key_falls_back_to_the_generic_rubric(self):
        rubric = get_rubric("does.not.exist", InterviewType.JAVA)
        assert rubric.key == "generic.java"

    def test_concept_lookup_climbs_to_the_nearest_ancestor(self):
        # No rubric authored for this leaf; spring.transactions is its parent.
        rubric = rubric_for_concept("spring.transactions.propagation", InterviewType.SPRING)
        assert rubric.key == "spring.transactions"

    def test_generic_rubric_still_has_dimensions(self):
        assert generic_rubric(InterviewType.CLOUD).dimensions


class TestRubricContent:
    def test_every_rubric_has_dimensions_and_probes(self):
        for rubric in RUBRICS:
            assert rubric.dimensions, f"{rubric.key} has no dimensions"
            for dimension in rubric.dimensions:
                assert dimension.hint, f"{rubric.key}.{dimension.key} has no probe question"

    def test_dimension_keys_are_unique_within_a_rubric(self):
        for rubric in RUBRICS:
            keys = rubric.dimension_keys()
            assert len(keys) == len(set(keys)), f"{rubric.key} has duplicate dimensions"

    def test_misconceptions_carry_a_correction_and_markers(self):
        for rubric in RUBRICS:
            for pattern in rubric.common_misconceptions:
                assert pattern.correction, f"{rubric.key}: '{pattern.belief}' has no correction"
                assert pattern.markers, f"{rubric.key}: '{pattern.belief}' has no markers"


def test_misconception_candidates_span_the_domain():
    """A Kafka question should also be able to catch a Kafka-ordering misconception."""
    rubric = get_rubric("kafka.delivery_semantics", InterviewType.DISTRIBUTED)
    patterns = misconception_candidates(["kafka.delivery_semantics"], rubric)
    beliefs = " ".join(pattern.belief.lower() for pattern in patterns)
    assert "ordering across the whole topic" in beliefs


def test_misconception_candidates_do_not_leak_across_domains():
    rubric = get_rubric("kafka.ordering", InterviewType.DISTRIBUTED)
    patterns = misconception_candidates(["kafka.ordering"], rubric)
    beliefs = " ".join(pattern.belief.lower() for pattern in patterns)
    assert "hashmap" not in beliefs


class TestAggregation:
    def _verdict(self, judge_key: str, score: float, **kwargs) -> JudgeVerdict:
        return JudgeVerdict(judge_key=judge_key, score=score, confidence=0.8, **kwargs)

    def test_weighted_across_judges(self):
        result = aggregate(
            [
                self._verdict("technical_accuracy", 1.0),
                self._verdict("reasoning", 0.0),
                self._verdict("hiring_bar", 0.0),
            ]
        )
        # technical carries 0.45 of the weight.
        assert result.score == pytest.approx(0.45, abs=0.01)

    def test_disagreement_lowers_confidence(self):
        agreeing = aggregate(
            [
                self._verdict("technical_accuracy", 0.8),
                self._verdict("reasoning", 0.8),
                self._verdict("hiring_bar", 0.8),
            ]
        )
        arguing = aggregate(
            [
                self._verdict("technical_accuracy", 1.0),
                self._verdict("reasoning", 0.1),
                self._verdict("hiring_bar", 0.9),
            ]
        )
        assert arguing.disagreement > agreeing.disagreement
        assert arguing.confidence < agreeing.confidence

    def test_communication_is_reported_separately_not_folded_into_the_score(self):
        result = aggregate(
            [
                self._verdict("technical_accuracy", 0.2),
                self._verdict("communication", 1.0, communication_score=1.0),
            ]
        )
        assert result.communication_score == 1.0
        # A beautifully explained wrong answer is still a wrong answer.
        assert result.score <= 0.3

    def test_a_misconception_caps_the_score(self):
        result = aggregate(
            [
                self._verdict("technical_accuracy", 0.95),
                self._verdict("reasoning", 0.95),
                self._verdict("hiring_bar", 0.95),
            ],
            MisconceptionFinding(
                detected=True,
                concept_key="kafka.ordering",
                belief="Kafka orders the whole topic.",
                correction="Ordering is per partition.",
            ),
        )
        assert result.score <= 0.45

    def test_missing_only_counts_when_no_judge_saw_it(self):
        result = aggregate(
            [
                self._verdict("technical_accuracy", 0.5, missing=["resize", "load_factor"]),
                self._verdict("reasoning", 0.5, demonstrated=["resize"]),
            ]
        )
        assert "resize" not in result.missing
        assert "load_factor" in result.missing

    def test_incorrect_is_unioned_across_judges(self):
        result = aggregate(
            [
                self._verdict("technical_accuracy", 0.4, incorrect=["a"]),
                self._verdict("reasoning", 0.4, incorrect=["b"]),
            ]
        )
        assert set(result.incorrect) == {"a", "b"}
