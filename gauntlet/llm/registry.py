"""Provider selection.

``get_provider()`` resolves the configured vendor, degrading to the deterministic offline
provider when credentials are missing. That degradation is logged loudly and reported
through ``/health`` so nobody mistakes heuristic scores for model scores.

Adding a vendor that speaks the OpenAI wire format needs no code here at all: it is a row
in :mod:`gauntlet.llm.providers.presets`.
"""

from __future__ import annotations

from functools import lru_cache

import structlog

from gauntlet.config import Settings, get_settings
from gauntlet.llm.base import LLMProvider
from gauntlet.llm.providers.presets import is_openai_compatible

log = structlog.get_logger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    resolved = settings.resolved_provider()

    if resolved != settings.llm_provider:
        preset = settings.preset_for()
        reason = (
            "no base URL configured"
            if preset.key == "custom"
            else f"{preset.api_key_env or 'credentials'} not set"
        )
        log.warning(
            "llm.provider.degraded",
            requested=settings.llm_provider,
            resolved=resolved,
            reason=reason,
            impact="interviews run on the offline heuristic engine, not a model",
        )

    if resolved == "anthropic":
        from gauntlet.llm.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)

    if is_openai_compatible(resolved):
        from gauntlet.llm.providers.openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(settings, settings.preset_for(resolved))

    from gauntlet.llm.providers.scripted import ScriptedProvider

    return ScriptedProvider()


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    return build_provider(get_settings())


def reset_provider_cache() -> None:
    get_provider.cache_clear()
