"""Agent base.

Every agent call goes through :meth:`Agent.invoke`, which is the single place where:

* a versioned prompt is rendered,
* trusted context is serialised into the ``<context>`` block,
* untrusted content is fenced with ``wrap_untrusted``,
* the response is validated into a Pydantic model,
* provenance (prompt name/version, model, provider) is captured.

Because it is one funnel, the security boundary and the reproducibility record cannot
be forgotten by an individual agent.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel

from gauntlet.llm.base import LLMProvider, StructuredResult
from gauntlet.llm.registry import get_provider
from gauntlet.prompts.registry import PromptTemplate, wrap_untrusted

log = structlog.get_logger(__name__)

# Untrusted text is truncated before it reaches a model: it bounds cost, and it bounds
# how much room an injection payload has to work with.
MAX_UNTRUSTED_CHARS = 24_000


class Agent:
    """Base class for every LLM-backed component."""

    key: str = "agent"

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider()
        return self._provider

    def invoke[T: BaseModel](
        self,
        template: PromptTemplate,
        response_model: type[T],
        *,
        context: dict[str, Any] | None = None,
        blocks: dict[str, str] | None = None,
        system_vars: dict[str, str] | None = None,
    ) -> StructuredResult[T]:
        """Render, call, validate.

        ``blocks`` maps an untrusted *source name* to its raw text. Each becomes the
        template variable ``$<source>_block``, fenced so the model treats it as data.
        """
        context_json = json.dumps(context or {}, default=str, ensure_ascii=False, indent=2)

        render_vars: dict[str, str] = {"context_json": context_json}
        for source, content in (blocks or {}).items():
            trimmed = (content or "")[:MAX_UNTRUSTED_CHARS]
            render_vars[f"{source}_block"] = wrap_untrusted(source, trimmed)

        system = template.render_system(**(system_vars or {}))
        user = template.render_user(**render_vars)

        log.debug(
            "agent.invoke",
            agent=self.key,
            prompt=template.name,
            version=template.version,
            model=response_model.__name__,
        )

        result = self.provider.complete_structured(
            system=system,
            user=user,
            response_model=response_model,
            role=template.role,
            temperature=template.temperature,
            max_tokens=template.max_tokens,
            prompt_name=template.name,
            prompt_version=template.version,
        )

        log.info(
            "agent.result",
            agent=self.key,
            prompt=f"{template.name}@v{template.version}",
            provider=result.provider,
            model=result.model,
            attempts=result.attempts,
            tokens=result.usage.total_tokens,
        )
        return result
