"""Shared test fixtures.

The whole suite runs against the deterministic offline provider, so tests assert on
interview *behaviour* rather than on a model's mood, cost nothing, and never flake on a
network call. Tests that genuinely need Postgres are marked and skipped when it is not
reachable, with a reason that says so.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

# Must be set before gauntlet.config is first imported.
os.environ.setdefault("GAUNTLET_LLM_PROVIDER", "scripted")
os.environ.setdefault("GAUNTLET_ENV", "test")
os.environ.setdefault("GAUNTLET_SECRET_KEY", "test-secret-not-used-in-production")
# Fail fast when Postgres is absent: the suite is designed to run without it, and a
# 5-second connect timeout per attempt turns skipped tests into a minute of waiting.
os.environ.setdefault("GAUNTLET_DB_CONNECT_TIMEOUT_SECONDS", "2")

from gauntlet.config import get_settings, reset_settings_cache
from gauntlet.db.session import database_available
from gauntlet.llm.embeddings import reset_embedder_cache
from gauntlet.llm.registry import get_provider, reset_provider_cache
from gauntlet.retrieval.question_index import reset_question_index
from gauntlet.skills.mastery import Evidence


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_db: needs a reachable Postgres instance")
    config.addinivalue_line(
        "markers", "requires_docker: needs a reachable Docker daemon for the sandbox"
    )


@pytest.fixture(autouse=True)
def _deterministic_environment() -> Iterator[None]:
    """Guarantee every test starts from the same provider and caches."""
    reset_settings_cache()
    reset_provider_cache()
    reset_embedder_cache()
    reset_question_index()
    yield
    reset_settings_cache()
    reset_provider_cache()
    reset_embedder_cache()
    reset_question_index()


@pytest.fixture
def provider():
    return get_provider()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def evidence_factory(now: datetime):
    def make(
        score: float,
        difficulty: int = 3,
        days_ago: float = 0.0,
        self_confidence: int | None = None,
        hints_used: int = 0,
        is_followup: bool = False,
    ) -> Evidence:
        from datetime import timedelta

        return Evidence(
            score=score,
            difficulty=difficulty,
            observed_at=now - timedelta(days=days_ago),
            self_confidence=self_confidence,
            hints_used=hints_used,
            is_followup=is_followup,
        )

    return make


@pytest.fixture(scope="session")
def db_available() -> bool:
    return database_available()


@pytest.fixture
def require_db(db_available: bool) -> None:
    if not db_available:
        pytest.skip(
            "Postgres is not reachable. Start it with: docker compose up -d db redis "
            "&& alembic upgrade head"
        )


RESUME_FIXTURE = """
Alex Morgan
Senior Backend Engineer

8 years of experience building Java backend services.

- Built Spring Boot microservices handling 40M requests per day.
- Implemented Kafka event processing pipeline for payment settlement.
- Reduced p99 API latency by 35% by introducing a Redis read-through cache.
- Optimised PostgreSQL queries and indexing for the reporting service.
"""

JOB_FIXTURE = """
Senior Java Engineer

Required:
- Strong Java 17+ and Spring Boot experience
- Kafka and event-driven architecture
- Strong SQL and PostgreSQL including indexing and query optimisation
- Designing and scaling microservices
"""

STRONG_HASHMAP_ANSWER = (
    "A HashMap computes the key's hashCode, then spreads the high bits into the low bits. "
    "The bucket index comes from (n-1) & hash because the table size is a power of two, "
    "so it is a mask rather than a modulo. On a collision entries form a linked list in "
    "the bucket and equals distinguishes keys. Long bins treeify into a red-black tree. "
    "Resize happens past capacity times the load factor of 0.75, rehashing into a table "
    "of double the size. Lookup is O(1) average case, O(log n) worst case."
)

CONFIDENTLY_WRONG_KAFKA = (
    "Kafka guarantees ordering across the topic, so consumers always see events in the "
    "order they were produced. That is the point of a log."
)
