"""Distributed tracing (spec section 43).

An interview is one request that fans out into a dozen model calls across several graph
nodes. When it is slow or expensive, per-call logs tell you *that* something took eleven
seconds and not *which* part of which turn. Spans nest, so they answer the question logs
cannot.

**Instrumentation is unconditional; the SDK is optional.** The OpenTelemetry *API* ships
a no-op tracer that costs approximately nothing when no SDK is installed, so the code path
is identical whether or not anyone is collecting. That avoids the usual mess of
``if tracing_enabled:`` scattered through business logic, and it means tracing cannot
change behaviour, only observe it.

Installing the collector side is opt-in:

    pip install -e ".[otel]"
    GAUNTLET_OTEL_ENABLED=true GAUNTLET_OTEL_ENDPOINT=http://localhost:4317

With nothing configured, every ``span()`` below is a no-op context manager.

Traces and logs are joined by ``add_trace_context``, a structlog processor that stamps the
active trace and span ids onto every log line. Without it you get two systems that both
describe the same incident and cannot be lined up.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

from gauntlet.llm.base import Usage
from gauntlet.observability.cost import estimate_cost

log = structlog.get_logger(__name__)

# Resolved through the global provider, which is a no-op until something installs an SDK.
tracer = trace.get_tracer("gauntlet")

_configured = False


def configure_tracing() -> bool:
    """Install the OpenTelemetry SDK when it is available and enabled.

    Returns whether tracing is actually collecting. Safe to call more than once, and
    designed never to raise: a broken collector must not stop interviews from running.
    """
    global _configured
    if _configured:
        return True

    from gauntlet.config import get_settings

    settings = get_settings()
    if not settings.otel_enabled:
        return False

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        log.warning(
            "tracing.sdk_missing",
            impact="tracing stays off; spans are created but nothing collects them",
            fix='pip install -e ".[otel]"',
        )
        return False

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": _version(),
            "deployment.environment": settings.env,
        }
    )
    provider = TracerProvider(resource=resource)

    exporter: Any
    if settings.otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
        except ImportError:
            log.warning(
                "tracing.exporter_missing",
                endpoint=settings.otel_endpoint,
                impact="falling back to console export",
            )
            exporter = ConsoleSpanExporter()
    else:
        # No endpoint configured but tracing asked for: print spans rather than
        # silently collecting nothing, which looks identical to tracing being broken.
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _configured = True
    log.info(
        "tracing.enabled",
        service=settings.otel_service_name,
        endpoint=settings.otel_endpoint or "console",
    )
    return True


def _version() -> str:
    from gauntlet import __version__

    return __version__


def tracing_active() -> bool:
    """Whether a real SDK is installed and collecting, as opposed to the no-op API."""
    return _configured


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    expected: tuple[type[BaseException], ...] = (),
) -> Iterator[Span]:
    """Start a span, recording exceptions and setting error status on the way out.

    Exceptions are recorded and re-raised, never swallowed: a span is a description of
    what happened, and a tracing layer that changes error handling is a liability.

    Attributes are an explicit dict rather than keyword arguments. OpenTelemetry
    attribute names are dotted and cannot be Python identifiers, so every call site was
    already spelling them out, and with ``**kwargs`` a key named "expected" would have
    been swallowed as the exception tuple.

    ``expected`` lists exception types that are control flow rather than failure. They
    are re-raised untouched and the span is left green. This is not a nicety: LangGraph
    pauses a graph by raising, so without it every interview marked a span as failed on
    every single turn, and a trace where normal operation is red is a trace nobody can
    find a real error in.
    """
    # The SDK records exceptions and sets error status by itself unless told not to.
    # Both are turned off and handled below, for two reasons: its automatic version
    # cannot know that a GraphInterrupt is a pause rather than a failure, and leaving it
    # on double-records every genuine error, once by the SDK and once here.
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as active:
        for key, value in (attributes or {}).items():
            if value is not None:
                active.set_attribute(key, value)
        try:
            yield active
        except expected:
            # Deliberate control flow. Recorded as a normal outcome, not an error.
            active.set_attribute("gauntlet.paused", True)
            raise
        except Exception as exc:
            active.record_exception(exc)
            active.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


def record_llm_call(
    active: Span,
    *,
    provider: str,
    model: str,
    usage: Usage,
    attempts: int = 1,
) -> float | None:
    """Attach model call detail to a span. Returns the estimated cost, if known.

    Follows the OpenTelemetry GenAI semantic conventions for the token and model
    attributes, so the spans mean something to tooling that was not written for this
    project.
    """
    active.set_attribute("gen_ai.system", provider)
    active.set_attribute("gen_ai.request.model", model)
    active.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    active.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    active.set_attribute("gauntlet.llm.attempts", attempts)

    cost = estimate_cost(model, usage)
    if cost is None:
        # Explicitly marked rather than left absent, so a dashboard can distinguish
        # "free" from "we do not know what this model costs".
        active.set_attribute("gauntlet.llm.cost_known", False)
    else:
        active.set_attribute("gauntlet.llm.cost_known", True)
        active.set_attribute("gauntlet.llm.cost_usd", cost)
    return cost


def add_trace_context(
    _logger: Any, _method: str, event: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor stamping trace and span ids onto every log line.

    This is what makes a log line and a span findable from each other. Without it,
    tracing and logging are two accounts of the same incident that cannot be joined.
    """
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event["trace_id"] = format(context.trace_id, "032x")
        event["span_id"] = format(context.span_id, "016x")
    return event


def current_trace_id() -> str | None:
    """The active trace id, for surfacing in an API response or error page."""
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None
