"""Provider-agnostic LLM interface (spec section 41).

Nothing above this layer knows which vendor is answering. Agents ask for a
*validated Pydantic model*, never a string, so a malformed generation fails loudly
at the boundary instead of silently corrupting an interview.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError

log = structlog.get_logger(__name__)


class LLMRole(StrEnum):
    """Which model tier a call should use. Interviewing wants the stronger model."""

    INTERVIEW = "interview"
    EVALUATION = "evaluation"


class LLMError(RuntimeError):
    """Provider call failed after exhausting retries."""


class StructuredOutputError(LLMError):
    """The model produced output that never validated against the target schema."""


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class StructuredResult[T: BaseModel]:
    """A validated model plus the provenance needed to reproduce it later."""

    value: T
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    prompt_name: str | None = None
    prompt_version: int | None = None
    attempts: int = 1


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that being strict here just
    produces avoidable failures. We try, in order: the whole string, a fenced block,
    then the outermost brace-balanced span.
    """
    candidates: list[str] = [text.strip()]

    fenced = _JSON_BLOCK.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise StructuredOutputError(f"no JSON object found in response: {text[:400]!r}")


def schema_instruction(response_model: type[BaseModel]) -> str:
    """The schema contract appended to prompts for providers without native tools."""
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    return (
        "Respond with a single JSON object and nothing else - no prose, no code fences.\n"
        f"It must validate against this JSON Schema:\n{schema}"
    )


class LLMProvider(ABC):
    """The contract every vendor adapter implements."""

    name: str = "abstract"

    @abstractmethod
    def model_for(self, role: LLMRole) -> str:
        """Concrete model id used for a given call tier - recorded on every evaluation."""

    @abstractmethod
    def complete_structured[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: LLMRole = LLMRole.INTERVIEW,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
    ) -> StructuredResult[T]:
        """Return a validated instance of ``response_model``."""

    @abstractmethod
    def complete_text(
        self,
        *,
        system: str,
        user: str,
        role: LLMRole = LLMRole.INTERVIEW,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        """Free-form completion. Used only where output is shown verbatim to a human."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts for retrieval."""

    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Score documents against a query, best first.

        Default is lexical overlap - a real cross-encoder is a provider-specific
        upgrade, but a deterministic baseline beats returning vector order unchanged.
        """
        query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
        scored: list[tuple[int, float]] = []
        for index, document in enumerate(documents):
            doc_terms = {
                term for term in re.findall(r"[a-z0-9]+", document.lower()) if len(term) > 2
            }
            if not query_terms or not doc_terms:
                scored.append((index, 0.0))
                continue
            overlap = len(query_terms & doc_terms)
            scored.append((index, overlap / len(query_terms | doc_terms)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class JSONModeProvider(LLMProvider):
    """Shared retry/validation loop for providers that return JSON as text.

    Subclasses only implement :meth:`_raw_completion`; parsing, schema-repair retries,
    and usage accounting live here so both vendor adapters behave identically.
    """

    max_attempts: int = 3

    @abstractmethod
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
        """Return (text, usage) from the vendor SDK."""

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        role: LLMRole = LLMRole.INTERVIEW,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        text, _ = self._raw_completion(
            system=system,
            user=user,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=None,
        )
        return text

    def complete_structured[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: LLMRole = LLMRole.INTERVIEW,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
    ) -> StructuredResult[T]:
        attempt_user = user
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                text, usage = self._raw_completion(
                    system=system,
                    user=attempt_user,
                    role=role,
                    temperature=temperature if attempt == 1 else 0.0,
                    max_tokens=max_tokens,
                    response_model=response_model,
                )
                payload = extract_json(text)
                value = response_model.model_validate(payload)
            except (StructuredOutputError, ValidationError) as exc:
                last_error = exc
                log.warning(
                    "llm.structured.retry",
                    provider=self.name,
                    model=response_model.__name__,
                    attempt=attempt,
                    error=str(exc)[:300],
                )
                # Feed the failure back so the next attempt can self-correct.
                attempt_user = (
                    f"{user}\n\nYour previous response was rejected: {str(exc)[:600]}\n"
                    "Return corrected JSON that satisfies the schema exactly."
                )
                continue
            except Exception as exc:  # vendor/network failure
                last_error = exc
                log.warning(
                    "llm.call.retry", provider=self.name, attempt=attempt, error=str(exc)[:300]
                )
                continue

            return StructuredResult(
                value=value,
                model=self.model_for(role),
                provider=self.name,
                usage=usage,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                attempts=attempt,
            )

        raise StructuredOutputError(
            f"{self.name} could not produce valid {response_model.__name__} "
            f"after {self.max_attempts} attempts: {last_error}"
        ) from last_error
