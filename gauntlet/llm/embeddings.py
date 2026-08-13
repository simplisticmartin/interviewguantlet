"""Embedding access, resolved independently of the chat provider.

That independence is not incidental. Several capable chat providers (DeepSeek, xAI,
Groq, Moonshot, Cerebras) expose no embedding endpoint at all, so tying embeddings to the
chat provider would mean losing retrieval the moment you switched models. Instead the
embedder resolves on its own: use the chat provider if it can embed, otherwise a provider
you nominate, otherwise a deterministic local fallback.

The fallback is explicitly *not* production-grade semantic search, and ``is_semantic``
lets callers say so rather than quietly pretending.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import structlog

from gauntlet.config import get_settings

log = structlog.get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9_]+")


class Embedder(ABC):
    is_semantic: bool = False
    dim: int = 1536
    backend: str = "unknown"

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
    backend = "local-hash"

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


class RemoteEmbedder(Embedder):
    """Any OpenAI-compatible embeddings endpoint."""

    is_semantic = True

    def __init__(self, *, backend: str, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI

        settings = get_settings()
        self.backend = backend
        self.dim = settings.embedding_dim
        self._model = model
        self._client = OpenAI(
            api_key=api_key or "not-required",
            base_url=base_url or None,
            timeout=settings.llm_timeout_seconds,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    preset, base_url, api_key, model = settings.resolve_embedding_choice()

    if preset is None:
        log.info(
            "embeddings.local",
            reason="no embedding-capable provider configured",
            impact="retrieval uses lexical hashing, not semantic similarity",
        )
        return HashingEmbedder(dim=settings.embedding_dim)

    try:
        embedder = RemoteEmbedder(
            backend=preset.key, base_url=base_url, api_key=api_key, model=model
        )
    except Exception as exc:  # pragma: no cover - depends on the SDK being installed
        log.warning("embeddings.degraded", provider=preset.key, error=str(exc)[:200])
        return HashingEmbedder(dim=settings.embedding_dim)

    log.info("embeddings.remote", provider=preset.key, model=model)
    return embedder


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
