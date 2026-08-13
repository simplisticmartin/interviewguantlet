"""``gauntlet-providers``: show what is supported and what is currently configured.

Discoverability matters here. The preset table is the answer to "can I use X?", and
making people read a source file to find out is a bad answer.
"""

from __future__ import annotations

import argparse

from gauntlet.config import get_settings
from gauntlet.llm.providers.presets import (
    COMPATIBLE_PRESETS,
    NATIVE_PRESETS,
    OFFLINE_PRESET,
    ProviderPreset,
)


def _status(preset: ProviderPreset) -> str:
    settings = get_settings()
    if preset.key == OFFLINE_PRESET.key:
        return "ready"
    if not preset.requires_key:
        return "ready" if settings.resolve_base_url(preset) else "needs base URL"
    return "key set" if settings.resolve_api_key(preset) else "no key"


def _row(preset: ProviderPreset, active: str) -> str:
    marker = " *" if preset.key == active else "  "
    embeddings = "yes" if preset.supports_embeddings else "no"
    return (
        f"{marker} {preset.key:<12} {preset.label:<34} "
        f"{_status(preset):<14} {embeddings:<11} {preset.api_key_env or '-'}"
    )


def _section(title: str, presets: tuple[ProviderPreset, ...], active: str) -> None:
    print(f"\n{title}")
    print("-" * 100)
    for preset in presets:
        print(_row(preset, active))


def main() -> None:
    parser = argparse.ArgumentParser(description="List Gauntlet model providers.")
    parser.add_argument("--verbose", action="store_true", help="Include notes and doc links.")
    args = parser.parse_args()

    settings = get_settings()
    active = settings.resolved_provider()

    print(f"\n{'':2} {'KEY':<12} {'PROVIDER':<34} {'STATUS':<14} {'EMBEDDINGS':<11} KEY VARIABLE")

    _section("Native adapters", NATIVE_PRESETS, active)
    _section("OpenAI-compatible", COMPATIBLE_PRESETS, active)
    _section("Offline", (OFFLINE_PRESET,), active)

    print(f"\nRequested : {settings.llm_provider}")
    print(f"Active    : {active}")
    if active != settings.llm_provider:
        print("            (degraded: credentials missing, running the offline engine)")

    preset = settings.preset_for(active)
    from gauntlet.llm.base import LLMRole

    print(f"Interview : {settings.resolve_model(preset, LLMRole.INTERVIEW)}")
    print(f"Evaluation: {settings.resolve_model(preset, LLMRole.EVALUATION)}")

    embed_preset, _, _, embed_model = settings.resolve_embedding_choice()
    if embed_preset is None:
        print("Embeddings: local hash fallback (not semantic)")
    else:
        print(f"Embeddings: {embed_preset.key} / {embed_model}")

    if args.verbose:
        print("\nNotes")
        print("-" * 100)
        for item in (*NATIVE_PRESETS, *COMPATIBLE_PRESETS):
            if item.notes or item.docs:
                print(f"\n{item.key} ({item.label})")
                if item.docs:
                    print(f"  docs: {item.docs}")
                if item.notes:
                    print(f"  {item.notes}")
    else:
        print("\nRun with --verbose for setup notes and documentation links.")

    print(
        "\nSet GAUNTLET_LLM_PROVIDER to a key above, plus that provider's key variable.\n"
        "Model names change often. Override with GAUNTLET_LLM_INTERVIEW_MODEL and\n"
        "GAUNTLET_LLM_EVALUATION_MODEL if a call reports an unknown model.\n"
    )


if __name__ == "__main__":
    main()
