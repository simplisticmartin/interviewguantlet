"""Question deduplication (spec section 8) and corpus recency (spec section 10).

The headline test is the spec's own example: three very differently worded versions of
Two Sum must collapse into one family, while genuinely different questions must not.
"""

from __future__ import annotations

import pytest

from gauntlet.content.questions import QUESTIONS
from gauntlet.ingestion.dedup import (
    QuestionCandidate,
    cluster_questions,
    compare,
    decayed_confidence,
    duplication_report,
    find_duplicates_of,
    jaccard,
    normalise,
    normalise_tokens,
    overlap_coefficient,
    recency_weight,
    signature,
)

# The spec's example, verbatim in spirit.
TWO_SUM_VARIANTS = [
    QuestionCandidate("a", "Find two numbers summing to target.", ("dsa.arrays",)),
    QuestionCandidate(
        "b",
        "Given an array and target, return two indices whose values add to target.",
        ("dsa.arrays",),
    ),
    QuestionCandidate("c", "Find a pair adding to K.", ("dsa.arrays",)),
]

DISTINCT = [
    QuestionCandidate(
        "d", "Reverse a linked list in place.", ("dsa.linked_lists",), ("linked_lists",)
    ),
    QuestionCandidate(
        "e", "Detect a cycle in a directed graph.", ("dsa.graphs",), ("graphs",)
    ),
]


class TestNormalisation:
    def test_synonyms_fold_to_one_token(self):
        assert "sum" in normalise_tokens("summing")
        assert "sum" in normalise_tokens("adds")
        assert "sum" in normalise_tokens("addition")

    def test_quantity_words_become_digits(self):
        assert "2" in normalise_tokens("two numbers")
        assert "2" in normalise_tokens("a pair")

    def test_stopwords_and_instruction_verbs_are_dropped(self):
        tokens = normalise_tokens("Walk me through how you would explain the array")
        assert "walk" not in tokens
        assert "explain" not in tokens
        assert "array" in tokens

    def test_plurals_fold(self):
        assert normalise_tokens("indices") == normalise_tokens("index")

    def test_normalise_is_stable_and_readable(self):
        assert normalise("Find two numbers summing to target.") == normalise(
            "find 2 numbers summing to target"
        )

    def test_signature_is_order_independent(self):
        assert signature("sum two numbers") == signature("two numbers sum")


class TestSimilarityMaths:
    def test_jaccard_bounds(self):
        assert jaccard({"a"}, {"a"}) == 1.0
        assert jaccard({"a"}, {"b"}) == 0.0
        assert jaccard(set(), {"a"}) == 0.0

    def test_overlap_coefficient_does_not_punish_a_terse_phrasing(self):
        short, long = {"sum", "2"}, {"sum", "2", "array", "index", "target", "return"}
        assert overlap_coefficient(short, long) == 1.0
        assert jaccard(short, long) < 0.5


class TestTheSpecExample:
    def test_all_three_two_sum_variants_collapse_into_one_family(self):
        clusters = cluster_questions(TWO_SUM_VARIANTS)
        assert len(clusters) == 1, [c.canonical.text for c in clusters]
        assert clusters[0].size == 3

    def test_the_most_complete_phrasing_becomes_canonical(self):
        clusters = cluster_questions(TWO_SUM_VARIANTS)
        assert "indices" in clusters[0].canonical.text

    def test_pairs_are_individually_recognised(self):
        for left, right in [(0, 1), (0, 2), (1, 2)]:
            score = compare(TWO_SUM_VARIANTS[left], TWO_SUM_VARIANTS[right])
            assert score.is_duplicate, (
                f"{TWO_SUM_VARIANTS[left].text} vs {TWO_SUM_VARIANTS[right].text}: "
                f"{score.explain()}"
            )


class TestNotOverMerging:
    def test_genuinely_different_questions_stay_apart(self):
        clusters = cluster_questions(DISTINCT)
        assert len(clusters) == 2

    def test_two_sum_does_not_absorb_unrelated_questions(self):
        clusters = cluster_questions([*TWO_SUM_VARIANTS, *DISTINCT])
        assert len(clusters) == 3
        assert max(cluster.size for cluster in clusters) == 3

    def test_concept_gate_blocks_merging_across_domains(self):
        """Similar wording, different subject, must not merge."""
        java = QuestionCandidate(
            "j", "How does indexing work and when is it slow?", ("java.collections.hashmap",)
        )
        database = QuestionCandidate(
            "p", "How does indexing work and when is it slow?", ("database.indexing",)
        )
        score = compare(java, database)
        assert score.lexical > 0.9, "wording is identical, so lexical should be high"
        assert not score.is_duplicate, "the concept gate should have blocked this"

    def test_adjacent_concepts_are_not_the_same_question(self):
        hashmap = QuestionCandidate(
            "h", "Walk me through a HashMap put.", ("java.collections.hashmap",)
        )
        chm = QuestionCandidate(
            "c",
            "Walk me through how ConcurrentHashMap handles concurrent writes.",
            ("java.concurrency.concurrent_hashmap",),
        )
        assert not compare(hashmap, chm).is_duplicate


class TestClusteringBehaviour:
    def test_transitive_merging(self):
        """A matches B and B matches C, so all three end up together."""
        clusters = cluster_questions(TWO_SUM_VARIANTS)
        assert len(clusters) == 1

    def test_empty_input(self):
        assert cluster_questions([]) == []

    def test_single_question_forms_its_own_family(self):
        clusters = cluster_questions([DISTINCT[0]])
        assert len(clusters) == 1
        assert not clusters[0].is_duplicate_group

    def test_slug_is_url_safe(self):
        clusters = cluster_questions(TWO_SUM_VARIANTS)
        slug = clusters[0].slug
        assert slug
        assert all(character.isalnum() or character == "-" for character in slug)

    def test_report_exposes_the_inflation_ratio(self):
        report = duplication_report(cluster_questions([*TWO_SUM_VARIANTS, *DISTINCT]))
        assert report["questions"] == 5
        assert report["families"] == 3
        assert report["questions_merged"] == 2
        assert report["inflation_ratio"] > 1.0

    def test_works_without_embeddings(self):
        """Normalisation alone must carry the Two Sum case, since offline embeddings
        are lexical and would not catch it."""
        clusters = cluster_questions(TWO_SUM_VARIANTS, use_embeddings=False)
        assert len(clusters) == 1


class TestIngestionCheck:
    def test_a_new_submission_matching_the_corpus_is_caught(self):
        corpus = list(TWO_SUM_VARIANTS)
        submission = QuestionCandidate(
            "new", "Return the indices of two values that sum to a target.", ("dsa.arrays",)
        )
        matches = find_duplicates_of(submission, corpus)
        assert matches, "an obvious reword should have been flagged"

    def test_a_genuinely_new_submission_is_accepted(self):
        submission = QuestionCandidate(
            "new", "Design a rate limiter for a public API.", ("system_design.rate_limiting",)
        )
        assert find_duplicates_of(submission, TWO_SUM_VARIANTS) == []


class TestShippedCorpus:
    def test_the_authored_corpus_has_no_accidental_duplicates(self):
        """The corpus is hand written, so any merge here is a real authoring mistake."""
        candidates = [
            QuestionCandidate(
                id=seed.slug,
                text=seed.question,
                concept_keys=tuple(seed.concept_keys),
                topics=tuple(seed.topics),
            )
            for seed in QUESTIONS
        ]
        clusters = cluster_questions(candidates)
        merged = [cluster for cluster in clusters if cluster.is_duplicate_group]
        assert not merged, [
            [member.id for member in cluster.members] for cluster in merged
        ]


class TestRecencyDecay:
    def test_fresh_evidence_keeps_full_weight(self):
        assert recency_weight(0) == 1.0

    def test_one_half_life_halves_it(self):
        assert recency_weight(365, half_life_days=365) == pytest.approx(0.5)

    def test_old_evidence_becomes_archival(self):
        assert recency_weight(365 * 7, half_life_days=365) < 0.01

    def test_more_evidence_raises_confidence(self):
        assert decayed_confidence(5, 0) > decayed_confidence(1, 0)

    def test_confidence_saturates_rather_than_growing_without_bound(self):
        assert decayed_confidence(100, 0) <= 1.0
        # The tenth report adds far less than the second.
        assert decayed_confidence(20, 0) - decayed_confidence(10, 0) < 0.1

    def test_recent_evidence_beats_a_larger_pile_of_stale_evidence(self):
        """Spec section 10: 2026 evidence outweighs 2015 evidence."""
        recent = decayed_confidence(3, age_days=30)
        stale = decayed_confidence(10, age_days=365 * 6)
        assert recent > stale

    def test_no_evidence_is_zero_confidence(self):
        assert decayed_confidence(0, 0) == 0.0
