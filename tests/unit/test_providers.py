"""Provider presets, credential resolution, and embedding decoupling.

No network. These test the resolution logic that decides which endpoint, model and key
get used, which is the part that silently does the wrong thing if it breaks.
"""

from __future__ import annotations

import pytest

from gauntlet.config import Settings
from gauntlet.llm.base import LLMRole
from gauntlet.llm.providers.presets import (
    ALL_PRESETS,
    COMPATIBLE_PRESETS,
    embedding_capable_keys,
    get_preset,
    is_openai_compatible,
    provider_keys,
)


def make_settings(**overrides: object) -> Settings:
    """Settings built from explicit values, ignoring any .env on disk."""
    base: dict[str, object] = {"llm_provider": "scripted", "embedding_provider": "local"}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


class TestPresetTable:
    def test_keys_are_unique(self):
        keys = [preset.key for preset in ALL_PRESETS]
        assert len(keys) == len(set(keys))

    def test_the_providers_people_actually_ask_for_are_present(self):
        expected = {
            "anthropic", "openai", "gemini", "deepseek", "xai", "moonshot", "qwen",
            "mistral", "groq", "together", "fireworks", "openrouter", "ollama",
        }
        assert expected <= set(provider_keys())

    def test_every_compatible_preset_has_a_base_url_or_is_explicitly_configurable(self):
        for preset in COMPATIBLE_PRESETS:
            if preset.key in {"openai", "custom", "azure"}:
                continue  # openai uses the SDK default, the others are user supplied
            assert preset.base_url, f"{preset.key} has no base URL"
            assert preset.base_url.startswith("http"), preset.key

    def test_every_preset_names_its_models(self):
        for preset in ALL_PRESETS:
            if preset.key == "custom":
                continue  # supplied by the user
            assert preset.interview_model, f"{preset.key} has no interview model"
            assert preset.evaluation_model, f"{preset.key} has no evaluation model"

    def test_every_preset_that_requires_a_key_names_the_variable(self):
        for preset in ALL_PRESETS:
            if preset.requires_key:
                assert preset.api_key_env, f"{preset.key} requires a key but names no env var"

    def test_anthropic_is_not_openai_compatible(self):
        assert not is_openai_compatible("anthropic")
        assert is_openai_compatible("deepseek")

    def test_providers_without_embeddings_are_marked_as_such(self):
        """Getting this wrong means retrieval silently breaks on those providers."""
        for key in ("deepseek", "xai", "groq", "moonshot", "cerebras"):
            preset = get_preset(key)
            assert preset is not None
            assert not preset.supports_embeddings, f"{key} does not have an embeddings API"

    def test_embedding_capable_list_is_accurate(self):
        capable = set(embedding_capable_keys())
        assert {"openai", "gemini", "mistral", "together", "ollama"} <= capable
        assert "deepseek" not in capable


class TestProviderValidation:
    def test_a_typo_fails_loudly_with_the_valid_options(self):
        with pytest.raises(ValueError, match="Unknown GAUNTLET_LLM_PROVIDER"):
            make_settings(llm_provider="deepsek")

    def test_provider_names_are_case_insensitive(self):
        assert make_settings(llm_provider="DeepSeek").llm_provider == "deepseek"

    def test_unknown_embedding_provider_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown GAUNTLET_EMBEDDING_PROVIDER"):
            make_settings(embedding_provider="nope")


class TestCredentialResolution:
    def test_missing_key_degrades_to_offline_rather_than_crashing(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        settings = make_settings(llm_provider="deepseek")
        assert settings.resolved_provider() == "scripted"

    def test_a_present_key_is_used(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        settings = make_settings(llm_provider="deepseek")
        assert settings.resolved_provider() == "deepseek"
        assert settings.resolve_api_key(settings.preset_for()) == "sk-test"

    def test_explicit_override_beats_the_preset_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
        settings = make_settings(llm_provider="deepseek", llm_api_key="from-override")
        assert settings.resolve_api_key(settings.preset_for()) == "from-override"

    def test_local_providers_need_no_key(self):
        settings = make_settings(llm_provider="ollama")
        assert settings.resolved_provider() == "ollama"

    def test_custom_without_a_base_url_degrades(self):
        assert make_settings(llm_provider="custom").resolved_provider() == "scripted"

    def test_custom_with_a_base_url_is_usable(self):
        settings = make_settings(
            llm_provider="custom",
            llm_base_url="https://gateway.internal/v1",
            llm_interview_model="my-model",
        )
        assert settings.resolved_provider() == "custom"
        assert settings.resolve_base_url(settings.preset_for()) == "https://gateway.internal/v1"

    def test_legacy_anthropic_and_openai_settings_still_work(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = make_settings(llm_provider="anthropic", anthropic_api_key="sk-legacy")
        assert settings.resolved_provider() == "anthropic"
        assert settings.resolve_api_key(settings.preset_for()) == "sk-legacy"


class TestModelResolution:
    def test_preset_defaults_are_used(self):
        settings = make_settings(llm_provider="deepseek")
        preset = settings.preset_for()
        assert settings.resolve_model(preset, LLMRole.INTERVIEW) == "deepseek-chat"

    def test_overrides_win(self):
        settings = make_settings(
            llm_provider="deepseek",
            llm_interview_model="deepseek-reasoner",
            llm_evaluation_model="deepseek-chat",
        )
        preset = settings.preset_for()
        assert settings.resolve_model(preset, LLMRole.INTERVIEW) == "deepseek-reasoner"
        assert settings.resolve_model(preset, LLMRole.EVALUATION) == "deepseek-chat"

    def test_interview_and_evaluation_tiers_are_separate(self):
        """Cheap model for grading, strong model for interviewing, is the point."""
        settings = make_settings(llm_provider="groq")
        preset = settings.preset_for()
        interview = settings.resolve_model(preset, LLMRole.INTERVIEW)
        evaluation = settings.resolve_model(preset, LLMRole.EVALUATION)
        assert interview != evaluation


class TestEmbeddingDecoupling:
    def test_a_chat_provider_without_embeddings_falls_back(self, monkeypatch):
        """The whole reason embeddings resolve separately."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = make_settings(llm_provider="deepseek", embedding_provider="auto")
        preset, _, _, _ = settings.resolve_embedding_choice()
        assert preset is None

    def test_embeddings_can_use_a_different_provider_than_chat(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-chat")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-embed")
        settings = make_settings(llm_provider="deepseek", embedding_provider="openai")
        preset, _, api_key, model = settings.resolve_embedding_choice()
        assert preset is not None
        assert preset.key == "openai"
        assert api_key == "sk-embed"
        assert model == "text-embedding-3-small"

    def test_auto_uses_the_chat_provider_when_it_can_embed(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        settings = make_settings(llm_provider="qwen", embedding_provider="auto")
        preset, _, _, model = settings.resolve_embedding_choice()
        assert preset is not None
        assert preset.key == "qwen"
        assert model == "text-embedding-v3"

    def test_local_forces_the_offline_embedder(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = make_settings(llm_provider="openai", embedding_provider="local")
        assert settings.resolve_embedding_choice()[0] is None

    def test_an_embedding_provider_without_a_key_falls_back(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        settings = make_settings(llm_provider="scripted", embedding_provider="mistral")
        assert settings.resolve_embedding_choice()[0] is None


class TestProviderConstruction:
    def test_the_offline_provider_is_always_buildable(self):
        from gauntlet.llm.registry import build_provider

        assert build_provider(make_settings()).name == "scripted"

    def test_a_compatible_provider_builds_without_calling_out(self, monkeypatch):
        """Constructing must not make a network request."""
        from gauntlet.llm.registry import build_provider

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        provider = build_provider(make_settings(llm_provider="deepseek"))
        assert provider.name == "deepseek"
        assert provider.model_for(LLMRole.INTERVIEW) == "deepseek-chat"

    @pytest.mark.parametrize(
        "key", ["gemini", "xai", "moonshot", "qwen", "groq", "openrouter", "together"]
    )
    def test_every_headline_provider_builds(self, monkeypatch, key: str):
        from gauntlet.llm.registry import build_provider

        preset = get_preset(key)
        assert preset is not None
        monkeypatch.setenv(preset.api_key_env, "sk-test")
        provider = build_provider(make_settings(llm_provider=key))
        assert provider.name == key
        assert provider.model_for(LLMRole.INTERVIEW)
        assert provider.model_for(LLMRole.EVALUATION)

    def test_local_providers_build_with_no_key(self):
        from gauntlet.llm.registry import build_provider

        assert build_provider(make_settings(llm_provider="ollama")).name == "ollama"
