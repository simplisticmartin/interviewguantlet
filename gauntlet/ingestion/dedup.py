"""Semantic question deduplication (spec section 8).

Interview questions arrive phrased a dozen different ways. These are the same question:

    "Find two numbers summing to target."
    "Given an array and target, return two indices whose values add to target."
    "Find a pair adding to K."

Without deduplication a corpus of 5,000 reports becomes a corpus of 5,000 "unique"
questions, most of which are the same forty problems reworded. That inflated count is
worse than useless: it makes the question bank look impressive while making retrieval
worse, because the same concept crowds out everything else in the results.

**How this works, and why it is not just embeddings.**

Vector similarity alone fails on exactly the cases that matter here. "HashMap" and
"ConcurrentHashMap" are semantically adjacent and are completely different questions,
while "find a pair adding to K" and "return two indices summing to target" are lexically
disjoint and are the same question. So three signals are combined:

1. **Normalised lexical overlap.** Domain-aware canonicalisation collapses the vocabulary
   the same problem gets described with: number words to digits, synonym families to one
   token, plurals folded. This is what catches the Two Sum case.
2. **Semantic similarity.** Embeddings, when a real embedding provider is configured.
   Degrades gracefully to lexical hashing offline, which is precisely why it is not the
   only signal.
3. **Concept overlap, used as a gate rather than a score.** Two questions tagged to
   disjoint concepts are never the same family regardless of how similar the words look.
   This is what stops the normaliser over-merging.

Clustering is union-find over pairs above threshold, so transitively related variants end
up in one family without needing every pair to match.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from gauntlet.llm.embeddings import cosine_similarity, get_embedder

# --- Tunables. Named, so they can be justified rather than guessed at. ---------
LEXICAL_WEIGHT = 0.55
SEMANTIC_WEIGHT = 0.45
JACCARD_WEIGHT = 0.4  # the remainder goes to overlap coefficient, see compare()
DUPLICATE_THRESHOLD = 0.72
# Below this concept overlap, two questions are never merged no matter how similar the
# wording. Prevents "explain X in Java" and "explain X in Postgres" collapsing.
CONCEPT_GATE = 0.30

_WORD = re.compile(r"[a-z0-9+#]+")

_STOPWORDS = frozenset(
    (
        "a an and are as at be by can could did do does for from give given has have how "
        "i if in into is it its me of on or so that the their them then there these this "
        "those to us was were what when which who why will with would you your walk "
        "through explain describe tell write implement design"
    ).split(" ")
)

# Synonym families. Every member maps to one canonical token, which is what lets three
# very differently worded versions of the same problem collapse together.
_SYNONYMS: dict[str, str] = {}


def _register(canonical: str, *variants: str) -> None:
    _SYNONYMS[canonical] = canonical
    for variant in variants:
        _SYNONYMS[variant] = canonical


# Quantities and structures
_register("2", "two", "pair", "couple", "double")
_register("3", "three", "triple", "triplet")
_register("k", "kth", "nth")
_register("array", "arrays", "list", "lists", "sequence", "collection", "nums", "numbers")
_register("string", "strings", "text", "chars", "characters", "substring", "substrings")
_register("index", "indices", "indexes", "position", "positions", "offset", "offsets")
_register("node", "nodes", "vertex", "vertices")
_register("edge", "edges", "link", "links")
_register("tree", "trees", "bst")
_register("graph", "graphs")
_register("map", "maps", "hashmap", "dictionary", "dict", "table")
_register("queue", "queues")
_register("stack", "stacks")
_register("cache", "caches", "caching")
# Operations
_register("sum", "sums", "summing", "add", "adds", "adding", "addition", "total", "totals")
_register("target", "goal", "value")
_register("find", "finds", "finding", "return", "returns", "locate", "get", "retrieve")
_register("max", "maximum", "largest", "biggest", "greatest", "highest")
_register("min", "minimum", "smallest", "lowest", "least")
_register("count", "counts", "counting", "number")
_register("sort", "sorts", "sorted", "sorting", "order", "ordered", "ordering")
_register("reverse", "reversed", "reversing")
_register("merge", "merged", "merging", "combine", "combined", "combining")
_register("duplicate", "duplicates", "repeated", "repeating", "repeat")
_register("unique", "distinct", "non-repeating", "nonrepeating")
_register("first", "earliest", "initial")
_register("longest", "largest-length")
_register("shortest", "smallest-length")
_register("complexity", "big-o", "runtime", "performance")
# Systems vocabulary
_register("scale", "scaling", "scales", "scalable")
_register("fail", "fails", "failure", "failures", "failing", "break", "breaks")
_register("thread", "threads", "threading", "concurrent", "concurrency")
_register("lock", "locks", "locking")
_register("partition", "partitions", "partitioning", "shard", "shards", "sharding")
_register("consumer", "consumers")
_register("producer", "producers")
_register("transaction", "transactions", "transactional")
_register("index-db", "indexing", "indexed")
_register("query", "queries", "querying")


def normalise_tokens(text: str) -> list[str]:
    """Canonical token sequence: lowercased, stopped, synonym folded, plural folded."""
    tokens: list[str] = []
    for raw in _WORD.findall(text.lower()):
        if raw in _STOPWORDS:
            continue
        # Single letters are variable placeholders, not content. "adding to K" and
        # "summing to target" describe the same problem, and K, N and X are arbitrary
        # names the author happened to pick. Multi-character identifiers such as c++
        # survive because the tokenizer keeps their punctuation.
        if len(raw) == 1 and raw.isalpha():
            continue
        token = _SYNONYMS.get(raw)
        if token is None:
            # Cheap plural folding for anything not in the synonym table.
            if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
                raw = raw[:-1]
            token = _SYNONYMS.get(raw, raw)
        if token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def normalise(text: str) -> str:
    """Human readable canonical form, useful for debugging a merge decision."""
    return " ".join(normalise_tokens(text))


def signature(text: str) -> frozenset[str]:
    return frozenset(normalise_tokens(text))


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coefficient(left: Iterable[str], right: Iterable[str]) -> float:
    """Overlap over the smaller set.

    Jaccard punishes a terse phrasing paired with a verbose one, which is the common case
    here ("find a pair adding to K" versus a full problem statement). Overlap coefficient
    does not, so the two are blended.
    """
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    """A question being considered for merging."""

    id: str
    text: str
    concept_keys: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()

    @property
    def tokens(self) -> frozenset[str]:
        return signature(self.text)


@dataclass(frozen=True, slots=True)
class SimilarityScore:
    lexical: float
    semantic: float
    concept: float
    combined: float
    is_duplicate: bool

    def explain(self) -> str:
        verdict = "duplicate" if self.is_duplicate else "distinct"
        return (
            f"{verdict}: combined={self.combined:.2f} "
            f"(lexical={self.lexical:.2f}, semantic={self.semantic:.2f}, "
            f"concept={self.concept:.2f})"
        )


def _with_ancestors(keys: Iterable[str]) -> set[str]:
    """Concept keys plus every dotted ancestor, so parent and child count as related."""
    expanded: set[str] = set()
    for key in keys:
        parts = key.split(".")
        for index in range(1, len(parts) + 1):
            expanded.add(".".join(parts[:index]))
    return expanded


def compare(
    left: QuestionCandidate,
    right: QuestionCandidate,
    *,
    embeddings: dict[str, list[float]] | None = None,
    threshold: float = DUPLICATE_THRESHOLD,
) -> SimilarityScore:
    """Score one pair. Concept overlap gates, it does not merely contribute."""
    # Overlap is weighted above Jaccard on purpose. The dominant real pattern is a terse
    # phrasing against a verbose one ("find a pair adding to K" versus a full problem
    # statement). Overlap asks "is the shorter one contained in the longer one?", which is
    # the duplicate question. Jaccard asks "are these the same length and content?", which
    # is stricter than what we mean and rejects exactly the pairs we want to catch.
    lexical = JACCARD_WEIGHT * jaccard(left.tokens, right.tokens) + (
        1 - JACCARD_WEIGHT
    ) * overlap_coefficient(left.tokens, right.tokens)

    semantic = 0.0
    has_semantic = False
    if embeddings is not None:
        left_vector = embeddings.get(left.id)
        right_vector = embeddings.get(right.id)
        if left_vector and right_vector:
            semantic = max(0.0, cosine_similarity(left_vector, right_vector))
            has_semantic = True

    # Untagged questions cannot be gated, so treat the gate as passed.
    if not left.concept_keys or not right.concept_keys:
        concept = 1.0
    else:
        # Expand with ancestors before comparing. Concept keys are a hierarchy, so
        # "kafka" and "kafka.ordering" are closely related, and comparing the raw
        # strings scores them zero. That matters most for contributed questions, which
        # get tagged coarsely and would otherwise never match a specifically tagged
        # corpus entry.
        concept = max(
            jaccard(_with_ancestors(left.concept_keys), _with_ancestors(right.concept_keys)),
            overlap_coefficient(
                _with_ancestors(left.concept_keys), _with_ancestors(right.concept_keys)
            ),
        )

    # Renormalise over the signals actually available. Without this, running with no
    # embedding provider silently scales every score by 0.55 and nothing ever merges,
    # which is a failure mode that looks like "dedup is just conservative" rather than
    # like a bug.
    combined = (
        LEXICAL_WEIGHT * lexical + SEMANTIC_WEIGHT * semantic if has_semantic else lexical
    )

    is_duplicate = combined >= threshold and concept >= CONCEPT_GATE

    return SimilarityScore(
        lexical=round(lexical, 4),
        semantic=round(semantic, 4),
        concept=round(concept, 4),
        combined=round(combined, 4),
        is_duplicate=is_duplicate,
    )


@dataclass
class QuestionFamilyCluster:
    """A group of questions judged to be the same underlying problem."""

    canonical: QuestionCandidate
    members: list[QuestionCandidate] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_duplicate_group(self) -> bool:
        return self.size > 1

    @property
    def slug(self) -> str:
        tokens = normalise_tokens(self.canonical.text)[:6]
        base = "-".join(tokens) or "question"
        return re.sub(r"[^a-z0-9-]", "", base)[:120]

    def topics(self) -> list[str]:
        seen: list[str] = []
        for member in self.members:
            for topic in member.topics:
                if topic not in seen:
                    seen.append(topic)
        return seen


class _UnionFind:
    def __init__(self, keys: Sequence[str]) -> None:
        self._parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self._parent[b] = a


def _embed_all(candidates: Sequence[QuestionCandidate]) -> dict[str, list[float]] | None:
    """Embeddings for the semantic signal, or None when there is no semantic signal.

    The offline fallback embedder is hashed bag-of-words: it reports `is_semantic = False`
    because it measures lexical overlap, not meaning. Feeding it in here would double
    count the lexical signal and then dilute it, which is worse than not having a semantic
    term at all. So this returns None unless the embedder genuinely understands meaning.
    """
    try:
        embedder = get_embedder()
        if not embedder.is_semantic:
            return None
        vectors = embedder.embed([candidate.text for candidate in candidates])
    except Exception:  # pragma: no cover - embedding is best effort here
        return None
    return {candidate.id: vector for candidate, vector in zip(candidates, vectors, strict=True)}


def cluster_questions(
    candidates: Sequence[QuestionCandidate],
    *,
    threshold: float = DUPLICATE_THRESHOLD,
    use_embeddings: bool = True,
) -> list[QuestionFamilyCluster]:
    """Group questions into families. Transitive by construction, via union-find."""
    if not candidates:
        return []

    embeddings = _embed_all(candidates) if use_embeddings else None
    union = _UnionFind([candidate.id for candidate in candidates])

    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if compare(left, right, embeddings=embeddings, threshold=threshold).is_duplicate:
                union.union(left.id, right.id)

    grouped: dict[str, list[QuestionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(union.find(candidate.id), []).append(candidate)

    clusters: list[QuestionFamilyCluster] = []
    for members in grouped.values():
        # The most complete phrasing becomes canonical: it carries the most constraints,
        # so it is the version worth keeping and showing.
        canonical = max(members, key=lambda item: (len(item.tokens), len(item.text)))
        clusters.append(QuestionFamilyCluster(canonical=canonical, members=members))

    clusters.sort(key=lambda cluster: (-cluster.size, cluster.canonical.id))
    return clusters


def duplication_report(clusters: Sequence[QuestionFamilyCluster]) -> dict[str, object]:
    """Summary suitable for logging or a CLI, including the honest inflation figure."""
    total = sum(cluster.size for cluster in clusters)
    families = len(clusters)
    duplicate_groups = [cluster for cluster in clusters if cluster.is_duplicate_group]
    return {
        "questions": total,
        "families": families,
        "duplicate_groups": len(duplicate_groups),
        "questions_merged": total - families,
        # How much a naive count would overstate the corpus.
        "inflation_ratio": round(total / families, 3) if families else 1.0,
        "largest_family": max((cluster.size for cluster in clusters), default=0),
    }


def find_duplicates_of(
    query: QuestionCandidate,
    corpus: Sequence[QuestionCandidate],
    *,
    threshold: float = DUPLICATE_THRESHOLD,
    limit: int = 5,
) -> list[tuple[QuestionCandidate, SimilarityScore]]:
    """Check one question against a corpus.

    This is the ingestion-time entry point: before accepting a newly contributed
    question, find out whether it is already in the bank under different words.
    """
    embeddings = _embed_all([query, *corpus])
    scored = [
        (candidate, compare(query, candidate, embeddings=embeddings, threshold=threshold))
        for candidate in corpus
        if candidate.id != query.id
    ]
    matches = [pair for pair in scored if pair[1].is_duplicate]
    matches.sort(key=lambda pair: pair[1].combined, reverse=True)
    return matches[:limit]


# ---------------------------------------------------------------------------
# Recency weighting (spec section 10)
# ---------------------------------------------------------------------------

DEFAULT_EVIDENCE_HALF_LIFE_DAYS = 365.0


def recency_weight(
    age_days: float, half_life_days: float = DEFAULT_EVIDENCE_HALF_LIFE_DAYS
) -> float:
    """Exponential decay for corpus evidence.

    Interview processes change. A report from this year is strong evidence, one from 2019
    is archival. Same principle as the mastery model, but a much longer half life: a
    company's interview shape moves far more slowly than a person's knowledge.
    """
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(age_days, 0.0) / half_life_days)


def decayed_confidence(
    evidence_count: int,
    age_days: float,
    *,
    half_life_days: float = DEFAULT_EVIDENCE_HALF_LIFE_DAYS,
    saturation: float = 4.0,
) -> float:
    """Confidence that a pattern still holds, from how much evidence and how old.

    Saturating rather than linear: the tenth report of the same thing adds far less than
    the second. Bounded to [0, 1] so it can be presented as a confidence directly.
    """
    if evidence_count <= 0:
        return 0.0
    weighted = evidence_count * recency_weight(age_days, half_life_days)
    return round(1.0 - math.exp(-weighted / saturation), 4)
