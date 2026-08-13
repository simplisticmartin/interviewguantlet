"""Anthropic adapter.

Structured output uses forced tool-use rather than "please return JSON", because a
tool schema is enforced by the API instead of by hope.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from gauntlet.config import Settings
from gauntlet.llm.base import JSONModeProvider, LLMRole, StructuredOutputError, Usage

_EMIT_TOOL = "emit_result"


class AnthropicProvider(JSONModeProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        from anthropic import Anthropic  # imported lazily: optional at runtime

        self._settings = settings
        self._client = Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self.max_attempts = max(1, settings.llm_max_retries)

    def model_for(self, role: LLMRole) -> str:
        if role is LLMRole.EVALUATION:
            return self._settings.anthropic_evaluation_model
        return self._settings.anthropic_interview_model

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
        kwargs: dict[str, Any] = {
            "model": self.model_for(role),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        if response_model is not None:
            schema = response_model.model_json_schema()
            kwargs["tools"] = [
                {
                    "name": _EMIT_TOOL,
                    "description": f"Emit a {response_model.__name__} result.",
                    "input_schema": _as_tool_schema(schema),
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": _EMIT_TOOL}

        response = self._client.messages.create(**kwargs)
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )

        if response_model is not None:
            for block in response.content:
                if getattr(block, "type", None) == "tool_use" and block.name == _EMIT_TOOL:
                    return json.dumps(block.input), usage
            raise StructuredOutputError("anthropic returned no tool_use block")

        parts = [
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ]
        return "\n".join(parts).strip(), usage

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Anthropic serves no embedding endpoint.

        Retrieval falls back to OpenAI embeddings when a key exists, otherwise the
        deterministic local embedder. Resolution happens in ``gauntlet.llm.embeddings``.
        """
        from gauntlet.llm.embeddings import get_embedder

        return get_embedder().embed(texts)


def _as_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs`` references so the schema is self-contained for the tools API."""
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 12:
            return node
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**resolve(target, depth + 1)}
                for key, value in node.items():
                    if key != "$ref":
                        merged[key] = resolve(value, depth + 1)
                return merged
            return {key: resolve(value, depth + 1) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item, depth + 1) for item in node]
        return node

    resolved = resolve(schema)
    if not isinstance(resolved, dict):  # pragma: no cover - defensive
        raise TypeError("resolved tool schema must be an object")
    return resolved


__all__ = ["AnthropicProvider"]
