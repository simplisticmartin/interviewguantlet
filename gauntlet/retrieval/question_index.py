"""Hybrid retrieval over the question corpus (spec section 12).

Pipeline, in order:

    query -> metadata filters -> BM25 lexical -> vector similarity -> fuse -> rerank

Not vector-only. Lexical retrieval matters here because interview questions hinge on
exact technical nouns - "ConcurrentHashMap", "REQUIRES_NEW", "p99" - and a purely
semantic match happily returns something adjacent but wrong.

This index serves the *live interview*, so it is in-process and deterministic: an
interview should not stall on a network round trip to pick the next question. The
question-browsing API (``/questions/search``) queries Postgres instead, over the same
corpus, where full-text and pgvector operators do the work at corpus scale.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from gauntlet.content.questions import QUESTIONS, QuestionSeed
from gauntlet.llm.embeddings import cosine_similarity, get_embedder
from gauntlet.schemas import InterviewType

_TOKEN = re.compile(r"[a-z0-9+#]+")
_STOP = frozenset(
    (
        "a an and are as at be by for from has have how in into is it its of on or that "
        "the to was were what when which who why with you your can could would will do "
        "does did"
    ).split(" ")
)

# BM25 parameters. Standard defaults; documented so they are tunable rather than folklore.
BM25_K1 = 1.5
BM25_B = 0.75

# Fusion weights: lexical carries more because technical nouns are the signal.
LEXICAL_WEIGHT = 0.6
VECTOR_WEIGHT = 0.4


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP]


@dataclass(frozen=True, slots=True)
class QuestionFilters:
    interview_types: frozenset[InterviewType] | None = None
    concept_keys: frozenset[str] | None = None
    min_difficulty: int = 1
    max_difficulty: int = 5
    level: str | None = None
    exclude_slugs: frozenset[str] = frozenset()
    expects_code: bool | None = None

    def accepts(self, seed: QuestionSeed) -> bool:
        if seed.slug in self.exclude_slugs:
            return False
        if self.interview_types is not None and seed.interview_type not in self.interview_types:
            return False
        if not self.min_difficulty <= seed.difficulty <= self.max_difficulty:
            return False
        if self.concept_keys is not None and not (set(seed.concept_keys) & self.concept_keys):
            return False
        if self.level is not None and seed.level is not None and seed.level != self.level:
            return False
        return self.expects_code is None or seed.expects_code == self.expects_code


@dataclass(frozen=True, slots=True)
class RetrievedQuestion:
    seed: QuestionSeed
    score: float
    lexical_score: float = 0.0
    vector_score: float = 0.0

    def as_context(self) -> dict[str, object]:
        return {
            "id": self.seed.slug,
            "prompt_text": self.seed.question,
            "interview_type": self.seed.interview_type.value,
            "agent_key": None,
            "concept_keys": list(self.seed.concept_keys),
            "difficulty": self.seed.difficulty,
            "rubric_key": self.seed.rubric_key,
            "expects_code": self.seed.expects_code,
            "asks_confidence": self.seed.asks_confidence,
            "follow_ups": list(self.seed.follow_ups),
            "retrieval_score": round(self.score, 4),
        }


@dataclass
class QuestionIndex:
    """BM25 + vector index over the seed corpus."""

    seeds: tuple[QuestionSeed, ...]
    _doc_tokens: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _doc_freq: dict[str, int] = field(default_factory=dict, repr=False)
    _avg_len: float = 0.0
    _embeddings: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        total_length = 0
        seen: dict[str, set[str]] = defaultdict(set)

        for seed in self.seeds:
            document = " ".join(
                [
                    seed.question,
                    " ".join(seed.topics),
                    " ".join(seed.concept_keys).replace(".", " "),
                ]
            )
            tokens = tokenize(document)
            self._doc_tokens[seed.slug] = tokens
            total_length += len(tokens)
            for token in set(tokens):
                seen[token].add(seed.slug)

        self._doc_freq = {token: len(slugs) for token, slugs in seen.items()}
        self._avg_len = total_length / len(self.seeds) if self.seeds else 0.0

    # --- Lexical ---------------------------------------------------------

    def _bm25(self, query_tokens: list[str], slug: str) -> float:
        tokens = self._doc_tokens.get(slug, [])
        if not tokens:
            return 0.0
        length = len(tokens)
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1

        total_docs = len(self.seeds)
        score = 0.0
        for token in query_tokens:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            doc_freq = self._doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            normalised_length = length / (self._avg_len or 1)
            denominator = frequency + BM25_K1 * (1 - BM25_B + BM25_B * normalised_length)
            score += idf * (frequency * (BM25_K1 + 1)) / denominator
        return score

    # --- Vector ----------------------------------------------------------

    def _embedding(self, seed: QuestionSeed) -> list[float]:
        cached = self._embeddings.get(seed.slug)
        if cached is None:
            cached = get_embedder().embed_one(f"{seed.question} {' '.join(seed.topics)}")
            self._embeddings[seed.slug] = cached
        return cached

    # --- Search ----------------------------------------------------------

    def search(
        self,
        query: str,
        filters: QuestionFilters | None = None,
        limit: int = 8,
        use_vectors: bool = True,
    ) -> list[RetrievedQuestion]:
        active = filters or QuestionFilters()
        candidates = [seed for seed in self.seeds if active.accepts(seed)]
        if not candidates:
            return []

        query_tokens = tokenize(query)
        lexical = {seed.slug: self._bm25(query_tokens, seed.slug) for seed in candidates}

        vector: dict[str, float] = dict.fromkeys(lexical, 0.0)
        if use_vectors and query.strip():
            query_vector = get_embedder().embed_one(query)
            for seed in candidates:
                vector[seed.slug] = max(0.0, cosine_similarity(query_vector, self._embedding(seed)))

        lexical_max = max(lexical.values(), default=0.0) or 1.0
        vector_max = max(vector.values(), default=0.0) or 1.0

        results = [
            RetrievedQuestion(
                seed=seed,
                score=(
                    LEXICAL_WEIGHT * (lexical[seed.slug] / lexical_max)
                    + VECTOR_WEIGHT * (vector[seed.slug] / vector_max)
                ),
                lexical_score=lexical[seed.slug],
                vector_score=vector[seed.slug],
            )
            for seed in candidates
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _variants_for(
        self,
        concept_keys: list[str],
        interview_type: InterviewType | None,
        exclude_slugs: frozenset[str],
        limit: int,
    ) -> list[RetrievedQuestion]:
        """Reframed versions of exhausted questions on the requested concepts.

        Only questions the candidate has already been asked are reframed: an unseen
        question is always better than a new frame on a seen one, and the earlier search
        steps have already established there are none left.
        """
        from gauntlet.content.variants import as_seed, pick_variant

        wanted = set(concept_keys)
        results: list[RetrievedQuestion] = []
        for seed in self.seeds:
            if seed.slug not in exclude_slugs:
                continue
            if wanted and not wanted.intersection(seed.concept_keys):
                continue
            if interview_type and seed.interview_type is not interview_type:
                continue
            variant = pick_variant(seed, seen_slugs=exclude_slugs)
            if variant is None:
                continue
            # Scored below anything real retrieval returns, so a variant is never
            # preferred over an actual unseen question.
            results.append(RetrievedQuestion(seed=as_seed(variant, seed), score=0.1))
            if len(results) >= limit:
                break
        return results

    def for_concepts(
        self,
        concept_keys: list[str],
        difficulty: int,
        interview_type: InterviewType | None = None,
        exclude_slugs: frozenset[str] = frozenset(),
        limit: int = 6,
    ) -> list[RetrievedQuestion]:
        """Interview-time entry point: nearest questions for a concept and difficulty.

        The difficulty window widens by one in each direction rather than demanding an
        exact match, because refusing to ask anything is worse than asking something
        slightly off-target.
        """
        from gauntlet.content.taxonomy import display_name

        query = " ".join(display_name(key) for key in concept_keys) or "software engineering"
        filters = QuestionFilters(
            interview_types=frozenset({interview_type}) if interview_type else None,
            concept_keys=frozenset(concept_keys) if concept_keys else None,
            min_difficulty=max(1, difficulty - 1),
            max_difficulty=min(5, difficulty + 1),
            exclude_slugs=exclude_slugs,
        )
        found = self.search(query, filters, limit=limit)
        if found:
            return found

        # Widen difficulty FIRST, keeping the concept. A question on the right topic at
        # the wrong difficulty beats an on-difficulty question about something else -
        # dropping the concept is what makes an interview feel incoherent.
        widened = self.search(
            query,
            QuestionFilters(
                interview_types=frozenset({interview_type}) if interview_type else None,
                concept_keys=frozenset(concept_keys) if concept_keys else None,
                exclude_slugs=exclude_slugs,
            ),
            limit=limit,
        )
        if widened:
            return widened

        # Before giving up the concept, try reframing a question on the right concept
        # that the candidate has already seen (spec section 39). On a fourth interview
        # the corpus runs dry concept by concept, and asking a known question in a new
        # frame keeps the interview coherent where switching concept does not.
        reframed = self._variants_for(concept_keys, interview_type, exclude_slugs, limit)
        if reframed:
            return reframed

        # Only now give up the concept, keeping type and difficulty band.
        return self.search(
            query,
            QuestionFilters(
                interview_types=frozenset({interview_type}) if interview_type else None,
                min_difficulty=max(1, difficulty - 1),
                max_difficulty=min(5, difficulty + 1),
                exclude_slugs=exclude_slugs,
            ),
            limit=limit,
        )


@lru_cache(maxsize=1)
def get_question_index() -> QuestionIndex:
    return QuestionIndex(seeds=QUESTIONS)


def reset_question_index() -> None:
    get_question_index.cache_clear()
