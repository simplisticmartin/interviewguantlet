"""The mastery model is the number the whole product rests on, so it is tested hard."""

from __future__ import annotations

import pytest

from gauntlet.skills.mastery import (
    Calibration,
    classify_calibration,
    compute_mastery,
    next_review,
    normalise_self_confidence,
    roll_up,
)


def test_no_evidence_is_zero_with_zero_confidence():
    state = compute_mastery([])
    assert state.mastery == 0.0
    assert state.confidence == 0.0
    assert state.calibration is Calibration.UNKNOWN


def test_mastery_is_not_a_plain_average(evidence_factory, now):
    """A hard success and an easy failure must not cancel to 0.5."""
    hard_success = compute_mastery([evidence_factory(1.0, difficulty=5)], now)
    easy_failure = compute_mastery([evidence_factory(0.0, difficulty=1)], now)
    assert hard_success.mastery > 0.95
    assert easy_failure.mastery == pytest.approx(0.0, abs=0.01)

    mixed = compute_mastery(
        [evidence_factory(1.0, difficulty=5), evidence_factory(0.0, difficulty=1)], now
    )
    # The difficulty-5 observation carries more weight, so the result sits above 0.5.
    assert mixed.mastery > 0.5


def test_recent_evidence_outweighs_stale_evidence(evidence_factory, now):
    improving = compute_mastery(
        [evidence_factory(0.2, days_ago=180), evidence_factory(0.9, days_ago=1)], now
    )
    declining = compute_mastery(
        [evidence_factory(0.9, days_ago=180), evidence_factory(0.2, days_ago=1)], now
    )
    assert improving.mastery > 0.6
    assert declining.mastery < 0.4


def test_followups_count_less_than_independent_answers(evidence_factory, now):
    independent = compute_mastery([evidence_factory(1.0), evidence_factory(1.0)], now)
    scaffolded = compute_mastery(
        [evidence_factory(1.0), evidence_factory(1.0, is_followup=True)], now
    )
    # Same scores, but a probe-scaffolded answer is weaker evidence, so we are less sure.
    assert scaffolded.confidence < independent.confidence


def test_hints_reduce_both_score_and_weight(evidence_factory, now):
    unaided = compute_mastery([evidence_factory(1.0)], now)
    hinted = compute_mastery([evidence_factory(1.0, hints_used=2)], now)
    assert hinted.mastery < unaided.mastery


def test_confidence_grows_with_consistent_evidence(evidence_factory, now):
    one = compute_mastery([evidence_factory(0.8)], now)
    several = compute_mastery([evidence_factory(0.8) for _ in range(5)], now)
    assert several.confidence > one.confidence


def test_disagreeing_evidence_lowers_confidence(evidence_factory, now):
    consistent = compute_mastery([evidence_factory(0.5) for _ in range(4)], now)
    erratic = compute_mastery(
        [
            evidence_factory(1.0),
            evidence_factory(0.0),
            evidence_factory(1.0),
            evidence_factory(0.0),
        ],
        now,
    )
    assert erratic.confidence < consistent.confidence


def test_mastery_is_bounded(evidence_factory, now):
    for score in (0.0, 0.5, 1.0):
        for difficulty in range(1, 6):
            state = compute_mastery([evidence_factory(score, difficulty=difficulty)], now)
            assert 0.0 <= state.mastery <= 1.0
            assert 0.0 <= state.confidence <= 1.0


class TestCalibration:
    """Spec section 22: the quadrants that drive study-plan priority."""

    def test_high_knowledge_high_confidence_is_mastery(self):
        assert classify_calibration(0.9, 0.9) is Calibration.MASTERY

    def test_low_knowledge_high_confidence_is_misconception(self):
        assert classify_calibration(0.2, 0.9) is Calibration.MISCONCEPTION

    def test_high_knowledge_low_confidence_is_a_confidence_deficit(self):
        assert classify_calibration(0.9, 0.1) is Calibration.CONFIDENCE_DEFICIT

    def test_low_knowledge_low_confidence_is_a_known_weakness(self):
        assert classify_calibration(0.2, 0.1) is Calibration.KNOWN_WEAKNESS

    def test_unrated_confidence_is_unknown(self):
        assert classify_calibration(0.9, None) is Calibration.UNKNOWN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1, 0.0), (3, 0.5), (5, 1.0), (None, None), (0, 0.0), (9, 1.0)],
)
def test_self_confidence_normalisation(raw, expected):
    assert normalise_self_confidence(raw) == expected


class TestSpacedRepetition:
    def test_weak_concepts_return_tomorrow(self, now):
        _, interval = next_review(0.2, previous_interval_days=8, now=now)
        assert interval == 1

    def test_strong_concepts_stretch_out(self, now):
        _, interval = next_review(0.9, previous_interval_days=8, now=now)
        assert interval > 8

    def test_intervals_are_capped(self, now):
        _, interval = next_review(0.95, previous_interval_days=1000, now=now)
        assert interval <= 120

    def test_due_date_follows_the_interval(self, now):
        due_at, interval = next_review(0.9, 4, now)
        assert (due_at - now).days == interval


def test_roll_up_weights_by_confidence(evidence_factory, now):
    confident_low = compute_mastery([evidence_factory(0.2) for _ in range(5)], now)
    uncertain_high = compute_mastery([evidence_factory(0.9)], now)
    parent = roll_up({"a": confident_low, "b": uncertain_high})
    # The well-evidenced weak concept should dominate the thinly-evidenced strong one.
    assert parent.mastery < 0.55
    assert parent.evidence_count == 6
