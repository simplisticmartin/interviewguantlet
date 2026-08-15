"""Contribution safety screening and pipeline (spec sections 37 and 38).

The safety tests carry the most weight here. A corpus that publishes somebody's
interviewer by name, or accepts NDA-covered material, is a legal and ethical problem
rather than a bug, so the bias is toward refusing and escalating.
"""

from __future__ import annotations

import pytest

from gauntlet.ingestion.pipeline import (
    Outcome,
    Submission,
    estimate_difficulty,
    infer_interview_type,
    process,
    tag_concepts,
)
from gauntlet.ingestion.safety import FindingKind, Verdict, redact, screen
from gauntlet.schemas import InterviewType


class TestRedaction:
    def test_emails_are_removed(self):
        report = screen("Ask me about Kafka, reachable at alex.morgan@example.com anytime.")
        assert "alex.morgan@example.com" not in report.text
        assert "[redacted]" in report.text
        assert FindingKind.EMAIL in report.kinds()

    def test_phone_numbers_are_removed(self):
        report = screen("They called me on +44 7700 900123 to schedule the round.")
        assert "900123" not in report.text
        assert FindingKind.PHONE in report.kinds()

    def test_urls_are_removed(self):
        report = screen("See https://linkedin.com/in/someone for the recruiter profile.")
        assert "linkedin.com" not in report.text
        assert FindingKind.URL in report.kinds()

    def test_social_handles_are_removed(self):
        report = screen("Follow @someone_real for more interview tips.")
        assert "@someone_real" not in report.text
        assert FindingKind.SOCIAL_HANDLE in report.kinds()

    def test_the_question_text_survives_redaction(self):
        """Scrubbing must not destroy the thing being contributed."""
        report = screen(
            "How does ConcurrentHashMap differ from a synchronized map? "
            "Reply to me@example.com"
        )
        assert "ConcurrentHashMap" in report.text
        assert "synchronized map" in report.text

    def test_redact_helper_returns_text_only(self):
        assert "@" not in redact("mail me at a@b.com")


class TestNameHandling:
    def test_an_interviewer_name_is_removed(self):
        report = screen("My interviewer was Sarah and she asked about Kafka partitions.")
        assert "Sarah" not in report.text
        assert FindingKind.PERSON_NAME in report.kinds()

    def test_a_named_recruiter_is_removed(self):
        report = screen("The recruiter named Daniel Brooks set up the screen.")
        assert "Daniel" not in report.text

    def test_a_signature_is_removed(self):
        report = screen("Great question about indexing.\n\nThanks, Priya")
        assert "Priya" not in report.text

    def test_a_removed_name_forces_human_review(self):
        """Removing the name may not be enough; the sentence can still identify someone."""
        report = screen("My interviewer was Sarah and she asked about Kafka.")
        assert report.verdict is Verdict.REVIEW

    @pytest.mark.parametrize(
        "text",
        [
            "Explain how Kafka handles partition rebalancing.",
            "Compare Redis and Postgres for a read heavy workload.",
            "What does Spring Boot do at startup?",
            "How does Java handle garbage collection?",
        ],
    )
    def test_capitalised_technology_names_are_never_redacted(self, text: str):
        """The failure mode that would make this filter useless."""
        report = screen(text)
        assert report.text == text
        assert FindingKind.PERSON_NAME not in report.kinds()
        assert report.verdict is Verdict.ACCEPT


class TestBlocking:
    def test_nda_material_is_refused(self):
        report = screen("This take-home was under NDA but here is the whole thing.")
        assert report.verdict is Verdict.BLOCK
        assert not report.accepted
        assert report.reasons

    def test_confidential_marking_is_refused(self):
        report = screen("Internal only, do not share: the full question set.")
        assert report.verdict is Verdict.BLOCK

    def test_leaked_assessment_is_refused(self):
        report = screen("Here is the leaked answer key from their online assessment.")
        assert report.verdict is Verdict.BLOCK
        assert FindingKind.LEAKED_ASSESSMENT in report.kinds()

    def test_blocking_short_circuits_redaction(self):
        """A blocked submission is refused, not scrubbed and kept."""
        report = screen("Confidential NDA material, contact me at a@b.com")
        assert report.verdict is Verdict.BLOCK
        assert report.redactions == 0

    def test_contact_solicitation_goes_to_review(self):
        report = screen("Great Kafka question. DM me if you want more of these.")
        assert report.verdict is Verdict.REVIEW

    def test_ordinary_content_is_accepted(self):
        report = screen("They asked me to design a rate limiter for a public API.")
        assert report.verdict is Verdict.ACCEPT
        assert report.findings == []


class TestConceptTagging:
    def test_tags_a_recognisable_concept(self):
        """Only surface forms that actually appear are tagged, so a coarse mention of
        Kafka tags the Kafka branch rather than guessing at the specific sub-concept."""
        keys = tag_concepts("What ordering guarantees does Kafka give across partitions?")
        assert any(key.startswith("kafka") for key in keys), keys

    def test_prefers_the_more_specific_concept(self):
        keys = tag_concepts("How does ConcurrentHashMap handle concurrent writes?")
        assert keys[0] == "java.concurrency.concurrent_hashmap"

    def test_returns_nothing_rather_than_guessing(self):
        """An untagged question is fine; a wrongly tagged one corrupts skill measurement."""
        assert tag_concepts("What is your favourite colour?") == []

    def test_interview_type_follows_the_concepts(self):
        keys = tag_concepts("Explain Kafka consumer group rebalancing.")
        assert infer_interview_type(keys) is InterviewType.DISTRIBUTED

    def test_difficulty_respects_a_stated_value(self):
        assert estimate_difficulty(["kafka.ordering"], 5) == 5

    def test_difficulty_falls_back_to_the_concept_floor(self):
        assert estimate_difficulty(["java.concurrency.memory_model"], None) >= 4


class TestPipeline:
    def test_a_good_submission_is_queued_not_published(self):
        """Nothing reaches the corpus without a human. That is the whole design."""
        result = process(
            Submission(
                question=(
                    "How would you make a payment API idempotent when the client retries "
                    "after a timeout and does not know if the charge succeeded?"
                ),
                company="stripe",
                level="senior",
            )
        )
        assert result.outcome is Outcome.QUEUED
        assert result.accepted
        assert result.concept_keys
        assert result.provenance["question_origin"] == "user_submitted"

    def test_provenance_never_claims_the_company_asks_it(self):
        result = process(
            Submission(
                question="Explain Kafka partition ordering guarantees.", company="google"
            )
        )
        assert "not evidence that this company asks this question" in (
            result.provenance["attribution_note"]
        )

    def test_safety_runs_before_anything_else(self):
        """An NDA submission must be refused before it is tagged or embedded."""
        result = process(
            Submission(question="Under NDA, but here is their full system design question set.")
        )
        assert result.outcome is Outcome.REJECTED
        assert result.concept_keys == []
        assert result.provenance == {}

    def test_a_reworded_existing_question_is_caught_as_a_duplicate(self):
        result = process(
            Submission(
                question=(
                    "What ordering guarantees does Kafka give you, and across what "
                    "scope do those guarantees apply?"
                )
            )
        )
        assert result.outcome is Outcome.DUPLICATE
        assert result.duplicate_of == "kafka-ordering-scope"
        assert result.duplicate_similarity is not None

    def test_a_close_but_uncertain_match_is_flagged_for_the_reviewer(self):
        """Between the near floor and the merge threshold, a human decides, not a number."""
        result = process(
            Submission(
                question=(
                    "What ordering guarantees does Kafka actually give you, and over "
                    "what scope do they hold?"
                )
            )
        )
        assert result.outcome is Outcome.QUEUED
        assert result.near_duplicates
        assert result.near_duplicates[0][0] == "kafka-ordering-scope"
        assert any("below the automatic merge threshold" in r for r in result.reasons)

    def test_personal_data_is_scrubbed_from_what_is_queued(self):
        result = process(
            Submission(
                question=(
                    "My interviewer was Sarah. She asked how I would design a distributed "
                    "rate limiter across twenty service instances."
                ),
                contributor_id="candidate-1",
            )
        )
        assert result.outcome is Outcome.QUEUED
        assert "Sarah" not in result.question
        assert result.safety.verdict is Verdict.REVIEW

    def test_too_short_is_rejected(self):
        assert process(Submission(question="kafka?")).outcome is Outcome.REJECTED

    def test_too_long_is_rejected(self):
        assert process(Submission(question="a" * 3000)).outcome is Outcome.REJECTED

    def test_an_untagged_question_is_queued_with_a_note(self):
        result = process(
            Submission(question="Tell me about a time the build broke on a Friday evening.")
        )
        assert result.outcome is Outcome.QUEUED
        if not result.concept_keys:
            assert any("manual tagging" in reason for reason in result.reasons)
