"""Provider selection.

``get_provider()`` resolves the configured vendor, degrading to the deterministic
offline provider when the required key is absent. That degradation is logged loudly
and reported through ``/health`` so nobody mistakes heuristic scores for model scores.
"""

from __future__ import annotations

from functools import lru_cache

import structlog

from gauntlet.config import Settings, get_settings
from gauntlet.llm.base import LLMProvider

log = structlog.get_logger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    resolved = settings.resolved_provider()

    if resolved != settings.llm_provider:
        log.warning(
            "llm.provider.degraded",
            requested=settings.llm_provider,
            resolved=resolved,
            reason="api key not configured",
        )

    if resolved == "anthropic":
        from gauntlet.llm.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if resolved == "openai":
        from gauntlet.llm.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)

    from gauntlet.llm.providers.scripted import ScriptedProvider

    return ScriptedProvider()


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    return build_provider(get_settings())


def reset_provider_cache() -> None:
    get_provider.cache_clear()
