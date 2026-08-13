"""Request-scoped dependencies: database session, authenticated candidate, limits."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from gauntlet.db.models import Candidate
from gauntlet.db.session import get_session_factory
from gauntlet.services.auth import AuthError, decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_db() -> Iterator[Session]:
    """One session per request, committed on success and rolled back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_current_candidate(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_db),
) -> Candidate:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_access_token(credentials.credentials)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    candidate_id = payload.get("cid")
    if not candidate_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")

    candidate = session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown candidate.")
    return candidate


# eq=False is load-bearing, not style: FastAPI keys its dependency cache by the callable,
# so a dependency object must be hashable. A plain dataclass defines __eq__, which sets
# __hash__ to None, and every endpoint using the limiter then fails with
# "unhashable type: RateLimiter". Identity semantics are what we want here anyway.
@dataclass(slots=True, eq=False)
class RateLimiter:
    """Fixed-window limiter, per client and route.

    In-process by design for a single-instance deployment. Running more than one API
    replica needs the Redis-backed variant - the limiter would otherwise allow
    `limit x replicas`. That swap is why the interface is a dependency.
    """

    limit: int
    window_seconds: float
    _hits: dict[str, deque[float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._hits = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        key = f"{client}:{request.url.path}"
        now = time.monotonic()
        window = self._hits[key]

        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.limit:
            retry_after = max(1, int(self.window_seconds - (now - window[0])))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Slow down.",
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)


# Auth endpoints are the ones worth protecting hardest: they are the credential-guessing
# surface. Uploads are throttled because parsing is comparatively expensive.
auth_rate_limit = RateLimiter(limit=10, window_seconds=60.0)
upload_rate_limit = RateLimiter(limit=20, window_seconds=60.0)
interview_rate_limit = RateLimiter(limit=120, window_seconds=60.0)
