"""Adapter for any OpenAI-compatible chat completions endpoint.

One class serves roughly twenty vendors, because most of them deliberately implement
OpenAI's wire format. The differences that actually matter are captured as data in
:mod:`gauntlet.llm.providers.presets`: base URL, model names, and which optional request
parameters the vendor will accept without erroring.

Structured output is requested through JSON mode where available and always reinforced
with an explicit schema contract in the system prompt, then validated through the shared
retry loop in :class:`JSONModeProvider`. That combination is the portable one: it works on
a frontier lab's API and on a 7B model running locally, which is the point of the
abstraction.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel

from gauntlet.config import Settings
from gauntlet.llm.base import JSONModeProvider, LLMRole, Usage, schema_instruction
from gauntlet.llm.providers.presets import ProviderPreset, get_preset

log = structlog.get_logger(__name__)


class OpenAICompatibleProvider(JSONModeProvider):
    """Talks to any endpoint implementing OpenAI chat completions."""

    def __init__(self, settings: Settings, preset: ProviderPreset) -> None:
        from openai import OpenAI  # imported lazily: the SDK is optional at runtime

        self._settings = settings
        self._preset = preset
        self.name = preset.key

        base_url = settings.resolve_base_url(preset)
        api_key = settings.resolve_api_key(preset)

        self._client = OpenAI(
            # Some local servers reject an empty key outright, so send a placeholder.
            api_key=api_key or "not-required",
            base_url=base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self.max_attempts = max(1, settings.llm_max_retries)
        self._json_mode = preset.supports_json_mode

        log.info(
            "llm.provider.ready",
            provider=preset.key,
            base_url=base_url or "default",
            interview_model=self.model_for(LLMRole.INTERVIEW),
            evaluation_model=self.model_for(LLMRole.EVALUATION),
            json_mode=self._json_mode,
        )

    def model_for(self, role: LLMRole) -> str:
        return self._settings.resolve_model(self._preset, role)

    def _raw_completion(
        self,
        *,
        system: str,
        user: str,
        role: LLMRole,
        temperature: float,
        max_tokens: int,
        response_model: type[BaseModel] | None,
    ) -> tuple[str, Usage]:
        system_prompt = system
        kwargs: dict[str, Any] = {}

        if response_model is not None:
            # The schema contract goes in the prompt regardless. JSON mode alone
            # guarantees valid JSON, not JSON of the right shape, and a good number of
            # these endpoints implement neither perfectly.
            system_prompt = f"{system}\n\n{schema_instruction(response_model)}"
            if self._json_mode:
                kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(
                model=self.model_for(role),
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except Exception as exc:
            # A provider that does not implement JSON mode usually rejects the whole
            # request rather than ignoring the parameter. Disable it and retry once,
            # permanently, so the rest of the interview does not keep paying for it.
            if self._json_mode and kwargs and _looks_like_unsupported_parameter(exc):
                log.warning(
                    "llm.json_mode.unsupported",
                    provider=self._preset.key,
                    detail=str(exc)[:200],
                    action="falling back to schema-in-prompt for this process",
                )
                self._json_mode = False
                return self._raw_completion(
                    system=system,
                    user=user,
                    role=role,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_model=response_model,
                )
            raise

        usage = Usage()
        if response.usage is not None:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )

        choice = response.choices[0].message
        content = choice.content or ""

        # Reasoning models on several providers put the answer in `content` but may
        # return an empty string when the response was entirely reasoning tokens.
        if not content.strip():
            reasoning = getattr(choice, "reasoning_content", None) or getattr(
                choice, "reasoning", None
            )
            if isinstance(reasoning, str):
                content = reasoning

        return content.strip(), usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate to the configured embedder.

        Chat and embeddings are resolved independently, because plenty of capable chat
        providers (DeepSeek, Grok, Groq, Moonshot) have no embedding endpoint at all.
        """
        from gauntlet.llm.embeddings import get_embedder

        return get_embedder().embed(texts)


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI proper. Kept as a named class because it is the reference implementation."""

    def __init__(self, settings: Settings) -> None:
        preset = get_preset("openai")
        if preset is None:  # pragma: no cover - the preset is defined in this package
            raise RuntimeError("openai preset missing")
        super().__init__(settings, preset)


_UNSUPPORTED_MARKERS = (
    "response_format",
    "unsupported parameter",
    "unrecognized request argument",
    "extra inputs are not permitted",
    "unknown field",
    "invalid_request_error",
)


def _looks_like_unsupported_parameter(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _UNSUPPORTED_MARKERS)
