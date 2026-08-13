"""Versioned prompt catalogue."""

from gauntlet.prompts.catalog import ALL_PROMPTS
from gauntlet.prompts.registry import (
    INJECTION_GUARD,
    REGISTRY,
    PromptRegistry,
    PromptTemplate,
    get_prompt,
    wrap_untrusted,
)

__all__ = [
    "ALL_PROMPTS",
    "INJECTION_GUARD",
    "REGISTRY",
    "PromptRegistry",
    "PromptTemplate",
    "get_prompt",
    "wrap_untrusted",
]
