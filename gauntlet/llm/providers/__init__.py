"""Vendor adapters.

Deliberately imports nothing heavy. ``gauntlet.config`` reads the preset table during
its own import, so pulling an adapter in here would create a cycle
(config -> presets package -> scripted -> embeddings -> config). Concrete adapters are
imported lazily at the point of use in :mod:`gauntlet.llm.registry`, which also keeps
the vendor SDKs optional at runtime.
"""

from gauntlet.llm.providers.presets import (
    ALL_PRESETS,
    COMPATIBLE_PRESETS,
    NATIVE_PRESETS,
    OFFLINE_PRESET,
    ProviderPreset,
    embedding_capable_keys,
    get_preset,
    is_openai_compatible,
    provider_keys,
)

__all__ = [
    "ALL_PRESETS",
    "COMPATIBLE_PRESETS",
    "NATIVE_PRESETS",
    "OFFLINE_PRESET",
    "ProviderPreset",
    "embedding_capable_keys",
    "get_preset",
    "is_openai_compatible",
    "provider_keys",
]
