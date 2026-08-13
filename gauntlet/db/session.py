"""Engine and session management.

A single lazily-created engine per process. Callers get sessions through
:func:`session_scope` (context manager, owns the transaction) or the FastAPI
dependency in ``apps.api.deps``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from gauntlet.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    connect_args: dict[str, object] = {}
    if settings.database_url.startswith("postgresql"):
        connect_args["connect_timeout"] = settings.db_connect_timeout_seconds

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
        connect_args=connect_args,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transaction boundary: commit on success, roll back on any exception."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_PROBE_CACHE_SECONDS = 10.0
_probe_result: tuple[float, bool] | None = None


def database_available(use_cache: bool = True) -> bool:
    """Connectivity probe for health checks and test skip-guards.

    Cached briefly so a health endpoint hammered while the database is down does not
    pay the connect timeout on every request.
    """
    global _probe_result

    if use_cache and _probe_result is not None:
        checked_at, result = _probe_result
        if time.monotonic() - checked_at < _PROBE_CACHE_SECONDS:
            return result

    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        result = True
    except Exception:
        result = False

    _probe_result = (time.monotonic(), result)
    return result


def reset_engine_cache() -> None:
    global _probe_result

    get_engine.cache_clear()
    get_session_factory.cache_clear()
    _probe_result = None
