"""API wiring tests that need no database.

These exist because of a real escape: `RateLimiter` was a plain dataclass, which made it
unhashable, which made FastAPI's dependency cache raise `TypeError` on *every*
rate-limited endpoint. Unit tests never touched HTTP and the database-backed API tests
skipped, so nothing caught it until the server was actually curled.

The lesson generalises: every route must be exercised through the ASGI stack at least
once, even with no database behind it. A dependency that cannot even be resolved is a
different class of bug from a query that fails.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.deps import auth_rate_limit, interview_rate_limit, upload_rate_limit
from apps.api.routers.contributions import contribution_rate_limit

# Endpoints that must resolve their dependencies without a database. Each is behind a
# rate limiter, which is exactly what broke.
RATE_LIMITED_POSTS = [
    ("/auth/register", {"email": "a@example.com", "password": "correct-horse-battery"}),
    ("/auth/login", {"email": "a@example.com", "password": "correct-horse-battery"}),
]


@pytest.fixture(scope="module")
def client():
    from apps.api.main import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


class TestDependencyWiring:
    def test_rate_limiters_are_hashable(self):
        """FastAPI keys its dependency cache by the callable object."""
        for limiter in (
            auth_rate_limit,
            upload_rate_limit,
            interview_rate_limit,
            contribution_rate_limit,
        ):
            assert hash(limiter) is not None
            assert {limiter: True}[limiter] is True

    @pytest.mark.parametrize(("path", "payload"), RATE_LIMITED_POSTS)
    def test_rate_limited_routes_resolve(self, client: TestClient, path: str, payload: dict):
        """A dependency-resolution failure shows up as 500 with no database involved."""
        response = client.post(path, json=payload)
        # 503 (no database) or 4xx (validation/auth) are all fine. 500 is not.
        assert response.status_code != 500, (
            f"{path} failed to resolve its dependencies: {response.text[:300]}"
        )

    def test_health_needs_no_database(self, client: TestClient):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "durable_checkpoints" in body

    def test_protected_routes_reject_anonymous_callers_before_touching_the_db(
        self, client: TestClient
    ):
        for path in (
            "/skills",
            "/analytics",
            "/study-plan",
            "/interviews",
            "/contributions/mine",
            "/moderation/submissions",
        ):
            assert client.get(path).status_code == 401, path

    def test_contributing_anonymously_is_rejected_before_any_screening(
        self, client: TestClient
    ):
        """Auth first: an unauthenticated payload should never reach the pipeline."""
        response = client.post("/contributions", json={"question": "a" * 40})
        assert response.status_code == 401

    def test_bulk_import_requires_auth_and_a_session(self, client: TestClient):
        """Bulk import writes rows, so it must resolve a database session.

        It originally took neither, which is how it managed to report questions as
        queued while storing nothing. A route that persists but never asks for a
        session is the shape of that bug.
        """
        response = client.post("/contributions/import", json={"payload": "[]"})
        assert response.status_code == 401

    def test_the_import_route_depends_on_the_database(self):
        """Checked structurally, since the behaviour needs Postgres to observe."""
        import inspect

        from apps.api.routers.contributions import bulk_import

        parameters = inspect.signature(bulk_import).parameters
        assert "session" in parameters, "bulk import cannot persist without a session"
        assert "candidate" in parameters, "imported rows need an owner"

    def test_companies_are_served_from_code_not_the_database(self, client: TestClient):
        """The catalogue is in-code, so it must work with no database at all."""
        response = client.get("/companies")
        assert response.status_code == 200
        assert len(response.json()) > 40

    def test_every_declared_route_is_reachable_in_the_schema(self, client: TestClient):
        spec = client.get("/openapi.json").json()
        for path in (
            "/auth/register",
            "/interviews",
            "/questions/search",
            "/analytics",
            "/contributions",
            "/moderation/submissions",
        ):
            assert path in spec["paths"], f"{path} missing from the OpenAPI schema"


class TestDatabaseOutageHandling:
    def test_a_database_outage_is_a_503_not_a_500(self, client: TestClient, db_available: bool):
        """An outage is not a bug in the request; the response should say what to do."""
        if db_available:
            pytest.skip("database is reachable, so this path cannot be exercised")

        response = client.post(
            "/auth/register",
            json={"email": "x@example.com", "password": "correct-horse-battery"},
        )
        assert response.status_code == 503
        assert "docker compose" in response.json()["detail"]
