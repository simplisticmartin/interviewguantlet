"""OpenAI adapter.

Uses JSON mode plus an explicit schema contract in the system prompt, then validates
through the shared retry loop in :class:`JSONModeProvider`. Also the source of real
embeddings when a key is configured.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from gauntlet.config import Settings
from gauntlet.llm.base import JSONModeProvider, LLMRole, Usage, schema_instruction


class OpenAIProvider(JSONModeProvider):
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # imported lazily: optional at runtime

        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self.max_attempts = max(1, settings.llm_max_retries)

    def model_for(self, role: LLMRole) -> str:
        if role is LLMRole.EVALUATION:
            return self._settings.openai_evaluation_model
        return self._settings.openai_interview_model

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
            system_prompt = f"{system}\n\n{schema_instruction(response_model)}"
            kwargs["response_format"] = {"type": "json_object"}

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

        usage = Usage()
        if response.usage is not None:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )
        return (response.choices[0].message.content or "").strip(), usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._settings.embedding_model, input=texts
        )
        return [item.embedding for item in response.data]
