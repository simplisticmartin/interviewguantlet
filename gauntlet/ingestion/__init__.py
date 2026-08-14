"""Corpus ingestion.

Currently holds deduplication and recency weighting, which are the two pieces the
question bank needs regardless of where questions come from. The source adapters,
extraction and moderation queue described in the roadmap build on top of these: dedup is
the gate a newly contributed question has to pass before it reaches production.
"""

from gauntlet.ingestion.dedup import (
    DUPLICATE_THRESHOLD,
    QuestionCandidate,
    QuestionFamilyCluster,
    SimilarityScore,
    cluster_questions,
    compare,
    decayed_confidence,
    duplication_report,
    find_duplicates_of,
    normalise,
    recency_weight,
)

__all__ = [
    "DUPLICATE_THRESHOLD",
    "QuestionCandidate",
    "QuestionFamilyCluster",
    "SimilarityScore",
    "cluster_questions",
    "compare",
    "decayed_confidence",
    "duplication_report",
    "find_duplicates_of",
    "normalise",
    "recency_weight",
]
