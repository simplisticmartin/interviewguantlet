"""Gauntlet API.

Run locally:

    uvicorn apps.api.main:app --reload
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from apps.api.routers import (
    auth,
    catalog,
    contributions,
    interviews,
    jobs,
    progress,
    resumes,
)
from apps.api.schemas import HealthResponse
from gauntlet import __version__
from gauntlet.config import get_settings
from gauntlet.db.session import database_available
from gauntlet.llm.embeddings import get_embedder
from gauntlet.observability import (
    add_trace_context,
    configure_tracing,
    cost_scope,
    current_trace_id,
    span,
    tracing_active,
)
from gauntlet.services.runtime import RUNTIME

log = structlog.get_logger(__name__)


def configure_logging() -> None:
    """Structured logs: JSON in production, human-readable in development."""
    settings = get_settings()
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        # Stamps trace and span ids so a log line and a span can be found from each
        # other. A no-op when nothing is tracing.
        add_trace_context,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if settings.is_production
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()

    configure_tracing()
    RUNTIME.start()

    if settings.resolved_provider() != settings.llm_provider:
        log.warning(
            "startup.llm_degraded",
            requested=settings.llm_provider,
            active=settings.resolved_provider(),
            impact="interviews run on the offline heuristic engine, not a model",
        )
    if settings.is_production and settings.secret_key.startswith("dev-secret"):
        raise RuntimeError("GAUNTLET_SECRET_KEY must be set to a real secret in production.")

    log.info(
        "startup",
        version=__version__,
        provider=settings.resolved_provider(),
        database=database_available(),
        durable_checkpoints=RUNTIME.durable_checkpoints,
        tracing=tracing_active(),
    )
    try:
        yield
    finally:
        RUNTIME.stop()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Gauntlet API",
        version=__version__,
        description=(
            "An adaptive, multi-agent technical interview simulator. "
            "Gauntlet is interview *preparation*: it does not assist with live "
            "employer assessments."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    app.include_router(auth.router)
    app.include_router(resumes.router)
    app.include_router(jobs.router)
    app.include_router(interviews.router)
    app.include_router(catalog.router)
    app.include_router(progress.router)
    app.include_router(contributions.router)

    @app.middleware("http")
    async def observe_request(request: Request, call_next):  # type: ignore[no-untyped-def]
        """One span per request, wrapping a cost scope for that request's model calls.

        The scope lives here rather than around the graph run so that anything a
        request triggers is counted, and so two concurrent requests cannot pool their
        spend into one number.
        """
        with (
            span(
                f"{request.method} {request.url.path}",
                **{
                    "http.request.method": request.method,
                    "url.path": request.url.path,
                },
            ) as active,
            cost_scope() as tally,
        ):
            response = await call_next(request)
            active.set_attribute("http.response.status_code", response.status_code)
            if tally.calls:
                active.set_attribute("gauntlet.cost.calls", tally.calls)
                active.set_attribute("gauntlet.cost.tokens", tally.total_tokens)
                active.set_attribute("gauntlet.cost.usd", tally.usd)
                # Says whether that figure is the whole story or a floor.
                active.set_attribute("gauntlet.cost.complete", tally.complete)
                log.info(
                    "request.cost",
                    path=request.url.path,
                    calls=tally.calls,
                    tokens=tally.total_tokens,
                    cost=tally.describe(),
                )
            trace_id = current_trace_id()
            if trace_id:
                # Lets a user quote one id from a failure and have it found in the trace
                # store, without exposing anything about internal structure.
                response.headers["X-Trace-Id"] = trace_id
            return response

    @app.exception_handler(OperationalError)
    async def database_unavailable(request: Request, exc: OperationalError) -> JSONResponse:
        # A database outage is not a bug in the request. Say so, with the fix, rather
        # than returning an opaque 500.
        log.error("request.database_unavailable", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": (
                    "The database is unavailable. Start it with "
                    "`docker compose up -d db redis` and apply migrations with "
                    "`alembic upgrade head`."
                )
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the detail, return a generic message: stack traces and driver errors are
        # not something an API should hand to a client.
        log.exception("request.failed", path=request.url.path, method=request.method)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error."},
        )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        resolved = settings.resolved_provider()
        return HealthResponse(
            status="ok",
            version=__version__,
            database=database_available(),
            llm_provider=resolved,
            # Surfaced so nobody mistakes heuristic scores for model-quality evaluation.
            llm_degraded=resolved != settings.llm_provider,
            durable_checkpoints=RUNTIME.durable_checkpoints,
            semantic_embeddings=get_embedder().is_semantic,
        )

    return app


app = create_app()
