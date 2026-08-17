"""Contribution and moderation over HTTP (spec sections 37, 38).

Needs Postgres, because the point of these tests is the persistence and authorisation
rules, which is exactly what the unit tests for the pipeline cannot cover:

    docker compose up -d db redis && alembic upgrade head && gauntlet-seed

The rules being defended are the ones that are embarrassing to get wrong. An
unauthorised account must not see the queue. A contributor must not be able to publish.
Raw personal data must not reach the database at all.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

pytestmark = pytest.mark.requires_db


@pytest.fixture
def client(require_db: None):
    from apps.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def register(client: TestClient) -> str:
    email = f"contrib-{uuid.uuid4().hex[:10]}@example.com"
    response = client.post(
        "/auth/register",
        json={"email": email, "password": "correct-horse-battery", "display_name": "C"},
    )
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


@pytest.fixture
def contributor(client: TestClient) -> TestClient:
    client.headers.update({"Authorization": f"Bearer {register(client)}"})
    return client


@pytest.fixture
def moderator(client: TestClient) -> TestClient:
    """A registered account promoted to moderator directly in the database."""
    from gauntlet.db.models import Candidate, User
    from gauntlet.db.session import get_session_factory

    token = register(client)
    client.headers.update({"Authorization": f"Bearer {token}"})

    me = client.get("/skills")
    assert me.status_code == 200

    with get_session_factory()() as session:
        # The newest candidate is the one just registered.
        candidate = session.execute(
            select(Candidate).order_by(Candidate.created_at.desc()).limit(1)
        ).scalar_one()
        user = session.get(User, candidate.user_id)
        assert user is not None
        user.is_moderator = True
        session.commit()

    return client


GOOD_QUESTION = (
    "How would you keep a payment API idempotent when a client retries after a timeout "
    "and cannot tell whether the original charge succeeded?"
)


class TestContributing:
    def test_a_submission_is_queued_and_says_it_is_not_published(
        self, contributor: TestClient
    ):
        response = contributor.post(
            "/contributions", json={"question": GOOD_QUESTION, "company": "stripe"}
        )
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "pending"
        assert "not published yet" in body["message"]

    def test_personal_data_never_reaches_the_database(self, contributor: TestClient):
        """The stored row must hold the redacted text, not what was sent."""
        response = contributor.post(
            "/contributions",
            json={
                "question": (
                    "My interviewer was Sarah, reach me at leaker@example.com. She asked "
                    "how I would shard a write heavy table across ten database nodes."
                )
            },
        )
        assert response.status_code == 202, response.text
        stored = response.json()["question"]
        assert "Sarah" not in stored
        assert "leaker@example.com" not in stored
        assert response.json()["safety_verdict"] == "review"

    def test_nda_material_is_refused_with_reasons(self, contributor: TestClient):
        response = contributor.post(
            "/contributions",
            json={
                "question": (
                    "This was under NDA but here is their entire system design question "
                    "set for senior candidates."
                )
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["reasons"]

    def test_a_contributor_sees_only_their_own_submissions(
        self, client: TestClient, contributor: TestClient
    ):
        contributor.post("/contributions", json={"question": GOOD_QUESTION})
        mine = contributor.get("/contributions/mine")
        assert mine.status_code == 200
        assert len(mine.json()) >= 1

        # A different account must not see them.
        client.headers.update({"Authorization": f"Bearer {register(client)}"})
        assert client.get("/contributions/mine").json() == []


class TestModerationAuthorisation:
    def test_an_ordinary_contributor_cannot_see_the_queue(self, contributor: TestClient):
        """404 rather than 403: the queue's existence is not confirmed to non-reviewers."""
        assert contributor.get("/moderation/submissions").status_code == 404

    def test_an_ordinary_contributor_cannot_rule_on_a_submission(
        self, contributor: TestClient
    ):
        created = contributor.post("/contributions", json={"question": GOOD_QUESTION})
        submission_id = created.json()["id"]
        response = contributor.post(
            f"/moderation/submissions/{submission_id}", json={"decision": "approve"}
        )
        assert response.status_code == 404

    def test_a_moderator_sees_the_queue(self, moderator: TestClient):
        assert moderator.get("/moderation/submissions").status_code == 200


class TestModerationDecisions:
    def test_approval_publishes_with_contributed_provenance(
        self, client: TestClient, moderator: TestClient
    ):
        created = moderator.post(
            "/contributions", json={"question": GOOD_QUESTION, "company": "stripe"}
        )
        assert created.status_code == 202, created.text
        submission_id = created.json()["id"]

        decided = moderator.post(
            f"/moderation/submissions/{submission_id}",
            json={
                "decision": "approve",
                "interview_type": "system_design",
                "concept_keys": ["system_design.idempotency"],
                "note": "Good question, tagged properly.",
            },
        )
        assert decided.status_code == 200, decided.text
        body = decided.json()
        assert body["status"] == "approved"
        assert body["published_question_id"]

        from gauntlet.db.models import Question
        from gauntlet.db.session import get_session_factory

        with get_session_factory()() as session:
            question = session.get(Question, uuid.UUID(body["published_question_id"]))
            assert question is not None
            # Contributed questions stay distinguishable from the authored corpus.
            assert question.question_origin == "user_submitted"
            assert question.source_type == "user_contribution"
            assert float(question.confidence) < 0.5

    def test_rejection_does_not_publish_anything(self, moderator: TestClient):
        created = moderator.post("/contributions", json={"question": GOOD_QUESTION})
        submission_id = created.json()["id"]

        decided = moderator.post(
            f"/moderation/submissions/{submission_id}",
            json={"decision": "reject", "note": "Too vague."},
        )
        assert decided.status_code == 200
        assert decided.json()["status"] == "rejected"
        assert decided.json()["published_question_id"] is None

    def test_a_submission_cannot_be_ruled_on_twice(self, moderator: TestClient):
        created = moderator.post("/contributions", json={"question": GOOD_QUESTION})
        submission_id = created.json()["id"]
        moderator.post(
            f"/moderation/submissions/{submission_id}",
            json={"decision": "reject", "note": "no"},
        )
        again = moderator.post(
            f"/moderation/submissions/{submission_id}",
            json={
                "decision": "approve",
                "interview_type": "system_design",
                "concept_keys": ["system_design.idempotency"],
            },
        )
        assert again.status_code == 409

    def test_publishing_untagged_is_refused(self, moderator: TestClient):
        """An untagged question in the corpus cannot be retrieved or scored properly."""
        created = moderator.post(
            "/contributions",
            json={"question": "Tell me about the hardest tradeoff you have argued for."},
        )
        assert created.status_code == 202
        if created.json()["concept_keys"]:
            pytest.skip("the tagger matched a concept, so this path does not apply")

        decided = moderator.post(
            f"/moderation/submissions/{created.json()['id']}", json={"decision": "approve"}
        )
        assert decided.status_code == 409


class TestBulkImportActuallyPersists:
    """Regression: the import endpoint reported success and wrote nothing.

    `import_payload` screened records and counted them, and the response told the
    contributor their questions were "queued for review", while nothing ever reached the
    database. That is worse than not having the feature: the notes are gone and the
    contributor has been told they are safe. The function is now named `preview` for what
    it does, and persisting lives in the service where a session exists.
    """

    NOTES = json.dumps(
        [
            {
                "question": (
                    "How would you migrate a live table to a new schema without "
                    "downtime or losing writes?"
                ),
                "company": "stripe",
            },
            {
                "question": (
                    "Explain how you would detect and recover from a poisoned message "
                    "stuck at the head of a queue."
                )
            },
        ]
    )

    def test_imported_questions_reach_the_review_queue(self, contributor: TestClient):
        response = contributor.post(
            "/contributions/import", json={"payload": self.NOTES}
        )
        assert response.status_code == 202, response.text
        assert response.json()["queued"] == 2

        # The claim in the response must be true of the database.
        mine = contributor.get("/contributions/mine").json()
        assert len(mine) >= 2

    def test_the_count_reported_matches_what_was_stored(self, contributor: TestClient):
        before = len(contributor.get("/contributions/mine").json())
        body = contributor.post(
            "/contributions/import", json={"payload": self.NOTES}
        ).json()
        after = len(contributor.get("/contributions/mine").json())
        assert after - before == body["queued"] + body["duplicates"]

    def test_refused_records_are_not_stored(self, contributor: TestClient):
        payload = json.dumps(
            [{"question": "This whole set is confidential and under NDA, but here it is."}]
        )
        body = contributor.post("/contributions/import", json={"payload": payload}).json()
        assert body["rejected"] == 1
        assert body["queued"] == 0

    def test_an_unknown_source_is_refused(self, contributor: TestClient):
        response = contributor.post(
            "/contributions/import", json={"payload": "[]", "source": "glassdoor"}
        )
        assert response.status_code == 422


class TestTheQueueDeduplicatesAgainstItself:
    """Regression: submissions were only compared against the authored corpus.

    A question fifty people were asked produced fifty identical pending rows, and a
    reviewer had to notice by hand. The queue is where duplicates cost the most, because
    each one spends a person's attention.
    """

    QUESTION = (
        "How would you design a system that deduplicates events arriving out of order "
        "from several producers at once?"
    )

    def test_the_same_question_submitted_twice_is_caught(self, contributor: TestClient):
        first = contributor.post("/contributions", json={"question": self.QUESTION})
        assert first.status_code == 202
        assert first.json()["status"] == "pending"

        second = contributor.post("/contributions", json={"question": self.QUESTION})
        assert second.status_code == 202
        assert second.json()["status"] == "duplicate", (
            "a question already sitting in the review queue was queued a second time"
        )

    def test_a_different_question_is_still_accepted(self, contributor: TestClient):
        contributor.post("/contributions", json={"question": self.QUESTION})
        other = contributor.post(
            "/contributions",
            json={
                "question": (
                    "Walk me through choosing a partition key for a table that is read "
                    "far more often than it is written."
                )
            },
        )
        assert other.json()["status"] == "pending"
