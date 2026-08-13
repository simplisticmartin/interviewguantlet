"""Evaluator regression tests (spec sections 20 and 47).

Thresholds are set at, or just under, what the current pipeline actually achieves - not
aspirationally. A prompt or rubric change that drops below them fails CI, which is the
whole point: evaluator quality should not be allowed to regress silently.

Measured baseline for the offline heuristic provider (2026-08):

    band accuracy            68%   <- systematically harsh; see note below
    ranking accuracy         92%
    misconception precision 100%
    misconception recall    100%
    false positive rate       0%

The gap between ranking and band accuracy is the honest characterisation of the offline
grader: it orders answers correctly but under-scores them, because matching surface
markers cannot see an idea expressed in unfamiliar words. Ranking is the metric that
matters for interview *steering* (harder vs easier), which is why the adaptive router
keys off relative score. Absolute calibration is what a real model provider buys you,
and this same benchmark measures that when one is configured.
"""

from __future__ import annotations

import pytest

from evals.runner import LABELS, load_datasets, run_benchmark

# Floors, not targets. Chosen from the measured baseline with a little headroom.
MIN_BAND_ACCURACY = 0.60
MIN_RANKING_ACCURACY = 0.85
MIN_MISCONCEPTION_PRECISION = 0.90
MIN_MISCONCEPTION_RECALL = 0.80
MAX_FALSE_POSITIVE_RATE = 0.10


@pytest.fixture(scope="module")
def report():
    return run_benchmark()


class TestDatasetIntegrity:
    def test_datasets_load(self):
        datasets = load_datasets()
        assert len(datasets) >= 5

    def test_every_case_has_a_known_label(self):
        for dataset in load_datasets():
            for case in dataset["cases"]:
                assert case["label"] in LABELS, f"{case['id']} has an unknown label"

    def test_every_topic_spans_the_quality_range(self):
        """A benchmark with only good answers measures nothing."""
        for dataset in load_datasets():
            labels = {case["label"] for case in dataset["cases"]}
            assert "excellent" in labels
            assert labels & {"wrong", "confidently_wrong"}

    def test_case_ids_are_unique(self):
        ids = [case["id"] for dataset in load_datasets() for case in dataset["cases"]]
        assert len(ids) == len(set(ids))

    def test_confidently_wrong_cases_expect_a_misconception(self):
        for dataset in load_datasets():
            for case in dataset["cases"]:
                if case["label"] == "confidently_wrong":
                    assert case.get("expects_misconception") is True


class TestEvaluatorQuality:
    def test_it_separates_the_best_answers_from_the_worst(self, report):
        """The floor. Failing this means the evaluator measures nothing at all."""
        assert report.separates_excellent_from_wrong(), (
            "the worst answer scored at least as high as the best - "
            "the evaluator is not discriminating"
        )

    def test_ranking_accuracy(self, report):
        assert report.ranking_accuracy >= MIN_RANKING_ACCURACY, (
            f"ranking accuracy {report.ranking_accuracy:.1%} fell below "
            f"{MIN_RANKING_ACCURACY:.0%}"
        )

    def test_band_accuracy(self, report):
        assert report.band_accuracy >= MIN_BAND_ACCURACY, (
            f"band accuracy {report.band_accuracy:.1%} fell below {MIN_BAND_ACCURACY:.0%}"
        )

    def test_misconception_precision(self, report):
        """A false accusation costs more trust than a miss."""
        assert report.misconception_precision >= MIN_MISCONCEPTION_PRECISION

    def test_misconception_recall(self, report):
        assert report.misconception_recall >= MIN_MISCONCEPTION_RECALL

    def test_false_positive_rate(self, report):
        assert report.false_positive_rate <= MAX_FALSE_POSITIVE_RATE

    def test_confidently_wrong_never_outscores_acceptable(self, report):
        """Fluent wrongness must not be rewarded (spec section 18)."""
        worst_acceptable = min(
            (case.score for case in report.cases if case.label == "acceptable"), default=1.0
        )
        best_confidently_wrong = max(
            (case.score for case in report.cases if case.label == "confidently_wrong"),
            default=0.0,
        )
        assert best_confidently_wrong < worst_acceptable or best_confidently_wrong <= 0.45

    def test_scores_stay_in_range(self, report):
        assert all(0.0 <= case.score <= 1.0 for case in report.cases)

    def test_the_run_is_deterministic_offline(self):
        """Same input, same score - otherwise regression testing is meaningless."""
        first = run_benchmark()
        second = run_benchmark()
        assert [case.score for case in first.cases] == [case.score for case in second.cases]
