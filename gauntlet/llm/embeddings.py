"""Embedding access.

Real embeddings come from OpenAI when a key is configured. Without one we fall back to
a deterministic hashed bag-of-words embedder so retrieval, dedup, and tests still run
offline. The fallback is explicitly *not* production-grade semantic search, and
``is_semantic`` lets callers say so in the UI rather than quietly pretending.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from gauntlet.config import get_settings

_TOKEN = re.compile(r"[a-z0-9_]+")


class Embedder(ABC):
    is_semantic: bool = False
    dim: int = 1536

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic hashed bag-of-words with sublinear term weighting.

    Good enough to make lexically-similar questions cluster (so dedup and retrieval are
    exercisable end to end) and completely offline. It has no semantic understanding:
    "car" and "automobile" are unrelated to it.
    """

    is_semantic = False

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vector

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


class OpenAIEmbedder(Embedder):
    is_semantic = True

    def __init__(self) -> None:
        from openai import OpenAI

        settings = get_settings()
        self.dim = settings.embedding_dim
        self._model = settings.embedding_model
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.openai_api_key:
        return OpenAIEmbedder()
    return HashingEmbedder(dim=settings.embedding_dim)


def reset_embedder_cache() -> None:
    get_embedder.cache_clear()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
