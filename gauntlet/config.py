"""Typed application configuration.

Everything is read from the environment (or `.env`) exactly once and exposed via
:func:`get_settings`. Nothing else in the codebase is allowed to read ``os.environ``
directly - that keeps configuration auditable and makes tests trivially overridable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "openai", "scripted"]

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    # Without an explicit timeout a half-open socket (a container that is starting, a
    # port-forward that accepts but never connects) blocks for minutes instead of
    # failing. Every connection attempt is bounded.
    db_connect_timeout_seconds: int = 5

    # --- LLM ----------------------------------------------------------------
    llm_provider: ProviderName = "scripted"
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    anthropic_interview_model: str = "claude-opus-5"
    anthropic_evaluation_model: str = "claude-sonnet-5"
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_interview_model: str = "gpt-4.1"
    openai_evaluation_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    llm_max_retries: int = 3
    llm_timeout_seconds: float = 90.0

    # --- Interview behaviour ------------------------------------------------
    default_interview_minutes: int = 20
    max_questions_per_interview: int = 14
    min_questions_per_interview: int = 4
    multi_judge_enabled: bool = True

    # --- Uploads ------------------------------------------------------------
    upload_dir: Path = REPO_ROOT / "uploads"
    max_upload_bytes: int = 5 * 1024 * 1024

    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, value: str) -> str:
        return value.strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def resolved_provider(self) -> ProviderName:
        """The provider we can actually use, given which keys are present.

        Falling back to ``scripted`` rather than raising means a fresh clone always
        boots: you get a deterministic offline interviewer until you add a key.
        """
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            return "scripted"
        if self.llm_provider == "openai" and not self.openai_api_key:
            return "scripted"
        return self.llm_provider


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Tests call this after mutating the environment."""
    get_settings.cache_clear()
