"""Typed application configuration.

Everything is read from the environment (or `.env`) exactly once and exposed via
:func:`get_settings`. Nothing else in the codebase reads ``os.environ`` directly - that
keeps configuration auditable and makes tests trivially overridable.

Provider configuration deliberately has three layers, checked in order:

1. An explicit override (``GAUNTLET_LLM_BASE_URL``, ``GAUNTLET_LLM_INTERVIEW_MODEL``,
   ``GAUNTLET_LLM_API_KEY``). Always wins, so any endpoint can be reached even if it has
   no preset.
2. The preset for the selected provider, which supplies sane defaults.
3. A fallback to the offline engine if no usable key is present, so a fresh clone still
   runs rather than crashing on startup.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gauntlet.llm.providers.presets import ProviderPreset

REPO_ROOT = Path(__file__).resolve().parent.parent

OFFLINE_PROVIDER = "scripted"


class Settings(BaseSettings):
    """Runtime configuration for the API, graph, and agents."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_prefix="GAUNTLET_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core ---------------------------------------------------------------
    env: Literal["development", "test", "production"] = "development"
    secret_key: str = "dev-secret-not-for-production"
    access_token_ttl_minutes: int = 60 * 12
    cors_origins: str = "http://localhost:5173"

    # --- Persistence --------------------------------------------------------
    database_url: str = "postgresql+psycopg://gauntlet:gauntlet@localhost:5433/gauntlet"
    redis_url: str = "redis://localhost:6380/0"
    db_connect_timeout_seconds: int = 5

    # --- LLM: which provider ------------------------------------------------
    # Any key from gauntlet.llm.providers.presets. Validated below so a typo fails fast
    # with the list of valid options rather than silently degrading to offline.
    llm_provider: str = OFFLINE_PROVIDER

    # Explicit overrides. These beat the preset, and make any endpoint reachable.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_interview_model: str = ""
    llm_evaluation_model: str = ""

    # Kept for backwards compatibility with the original two-provider configuration.
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    anthropic_interview_model: str = ""
    anthropic_evaluation_model: str = ""
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_interview_model: str = ""
    openai_evaluation_model: str = ""

    llm_max_retries: int = 3
    llm_timeout_seconds: float = 90.0

    # --- LLM: embeddings, resolved independently ----------------------------
    # "auto" means use the chat provider when it supports embeddings, otherwise fall
    # back to the deterministic local embedder. Set explicitly to mix providers, for
    # example chat on DeepSeek and embeddings on OpenAI.
    embedding_provider: str = "auto"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1536

    # --- Interview behaviour ------------------------------------------------
    default_interview_minutes: int = 20
    max_questions_per_interview: int = 14
    min_questions_per_interview: int = 4
    multi_judge_enabled: bool = True

    # --- Uploads ------------------------------------------------------------
    upload_dir: Path = REPO_ROOT / "uploads"
    max_upload_bytes: int = 5 * 1024 * 1024

    # --- Validation ---------------------------------------------------------

    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, value: str) -> str:
        return value.strip()

    @field_validator("llm_provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        from gauntlet.llm.providers.presets import get_preset, provider_keys

        normalised = value.strip().lower()
        if get_preset(normalised) is None:
            raise ValueError(
                f"Unknown GAUNTLET_LLM_PROVIDER '{value}'. "
                f"Valid options: {', '.join(provider_keys())}. "
                "Use 'custom' with GAUNTLET_LLM_BASE_URL for anything else."
            )
        return normalised

    @field_validator("embedding_provider")
    @classmethod
    def _known_embedding_provider(cls, value: str) -> str:
        from gauntlet.llm.providers.presets import get_preset

        normalised = value.strip().lower()
        if normalised in {"auto", "local", "none"}:
            return normalised
        if get_preset(normalised) is None:
            raise ValueError(
                f"Unknown GAUNTLET_EMBEDDING_PROVIDER '{value}'. "
                "Use 'auto', 'local', or a provider key."
            )
        return normalised

    # --- Derived ------------------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    # --- Provider resolution ------------------------------------------------

    def preset_for(self, key: str | None = None) -> ProviderPreset:
        from gauntlet.llm.providers.presets import get_preset

        preset = get_preset(key or self.llm_provider)
        if preset is None:  # pragma: no cover - validator prevents this
            raise ValueError(f"unknown provider: {key or self.llm_provider}")
        return preset

    def resolve_api_key(self, preset: ProviderPreset) -> str:
        """Explicit override, then legacy fields, then the preset's environment variable."""
        if self.llm_api_key:
            return self.llm_api_key
        if preset.key == "anthropic" and self.anthropic_api_key:
            return self.anthropic_api_key
        if preset.key == "openai" and self.openai_api_key:
            return self.openai_api_key
        if not preset.api_key_env:
            return ""
        # The only place outside pydantic that touches the environment, because the set
        # of provider key variables is open ended and cannot be declared as fields.
        return os.environ.get(preset.api_key_env, "").strip()

    def resolve_base_url(self, preset: ProviderPreset) -> str:
        return (self.llm_base_url or preset.base_url or "").strip()

    def resolve_model(self, preset: ProviderPreset, role: object) -> str:
        """Model for a call tier. Overrides beat legacy fields, which beat the preset."""
        from gauntlet.llm.base import LLMRole

        is_evaluation = role is LLMRole.EVALUATION

        if is_evaluation and self.llm_evaluation_model:
            return self.llm_evaluation_model
        if not is_evaluation and self.llm_interview_model:
            return self.llm_interview_model

        if preset.key == "anthropic":
            legacy = (
                self.anthropic_evaluation_model if is_evaluation
                else self.anthropic_interview_model
            )
            if legacy:
                return legacy
        if preset.key == "openai":
            legacy = (
                self.openai_evaluation_model if is_evaluation else self.openai_interview_model
            )
            if legacy:
                return legacy

        return preset.evaluation_model if is_evaluation else preset.interview_model

    def has_usable_credentials(self, preset: ProviderPreset) -> bool:
        """Whether this provider could actually be called."""
        if preset.key == OFFLINE_PROVIDER:
            return True
        if not preset.requires_key:
            # Local servers need an address rather than a key.
            return bool(self.resolve_base_url(preset))
        return bool(self.resolve_api_key(preset))

    def resolved_provider(self) -> str:
        """The provider we can actually use, given what is configured.

        Falling back to the offline engine rather than raising means a fresh clone always
        boots. The degradation is logged and surfaced through ``/health`` so nobody
        mistakes heuristic scores for model scores.
        """
        preset = self.preset_for()
        if preset.key == OFFLINE_PROVIDER:
            return OFFLINE_PROVIDER
        if preset.key == "custom" and not self.resolve_base_url(preset):
            return OFFLINE_PROVIDER
        return preset.key if self.has_usable_credentials(preset) else OFFLINE_PROVIDER

    def resolve_embedding_choice(self) -> tuple[ProviderPreset | None, str, str, str]:
        """Return (preset, base_url, api_key, model) for embeddings, or None to go local.

        Embeddings are resolved separately from chat because several strong chat
        providers have no embedding endpoint at all.
        """
        if self.embedding_provider in {"local", "none"}:
            return None, "", "", ""

        if self.embedding_provider == "auto":
            chat = self.preset_for(self.resolved_provider())
            candidate = chat if chat.supports_embeddings else None
            # OpenAI is the common secondary when the chat provider cannot embed.
            if candidate is None and self.openai_api_key:
                from gauntlet.llm.providers.presets import get_preset

                candidate = get_preset("openai")
        else:
            candidate = self.preset_for(self.embedding_provider)

        if candidate is None or not candidate.supports_embeddings:
            return None, "", "", ""

        api_key = self.embedding_api_key or self.resolve_api_key(candidate)
        if candidate.requires_key and not api_key:
            return None, "", "", ""

        base_url = self.embedding_base_url or candidate.base_url or ""
        model = self.embedding_model or candidate.embedding_model or ""
        if not model:
            return None, "", "", ""
        return candidate, base_url, api_key, model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests call this after mutating the environment."""
    get_settings.cache_clear()
