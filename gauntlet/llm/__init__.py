"""LLM access layer: one interface, swappable vendors."""

from gauntlet.llm.base import (
    JSONModeProvider,
    LLMError,
    LLMProvider,
    LLMRole,
    StructuredOutputError,
    StructuredResult,
    Usage,
)
from gauntlet.llm.embeddings import Embedder, cosine_similarity, get_embedder
from gauntlet.llm.registry import get_provider, reset_provider_cache

__all__ = [
    "Embedder",
    "JSONModeProvider",
    "LLMError",
    "LLMProvider",
    "LLMRole",
    "StructuredOutputError",
    "StructuredResult",
    "Usage",
    "cosine_similarity",
    "get_embedder",
    "get_provider",
    "reset_provider_cache",
]
