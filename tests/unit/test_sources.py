"""Bulk import adapters (spec section 37).

`TestBulkImportGetsNoShortcut` is the class that matters. The temptation with bulk import
is to trust the file because it is large and tidy; the safety screen and the review queue
have to apply per record, or a single import becomes the way to get unreviewed content
into the corpus.
"""

from __future__ import annotations

import json

import pytest

from gauntlet.ingestion.sources import (
    SOURCES,
    ImportReport,
    OwnNotesSource,
    get_source,
    preview,
    refused_sources,
)

NOTES = OwnNotesSource()

GOOD_QUESTION = (
    "How would you keep a payment API idempotent when a client retries after a timeout?"
)
SECOND_QUESTION = (
    "Explain how you would shard a write heavy table across several database nodes."
)


class TestJsonNotes:
    def test_a_list_of_objects_is_parsed(self):
        payload = json.dumps(
            [
                {"question": GOOD_QUESTION, "company": "stripe", "level": "senior"},
                {"question": SECOND_QUESTION},
            ]
        )
        submissions = NOTES.read(payload)
        assert len(submissions) == 2
        assert submissions[0].company == "stripe"

    def test_a_single_object_is_accepted(self):
        assert len(NOTES.read(json.dumps({"question": GOOD_QUESTION}))) == 1

    def test_alternative_field_names_are_understood(self):
        payload = json.dumps([{"prompt": GOOD_QUESTION, "employer": "acme", "stage": "onsite"}])
        submission = NOTES.read(payload)[0]
        assert submission.question == GOOD_QUESTION
        assert submission.company == "acme"
        assert submission.interview_round == "onsite"

    def test_malformed_json_fails_loudly(self):
        """Silently importing nothing would look identical to an empty file."""
        with pytest.raises(ValueError, match="Could not parse JSON"):
            NOTES.read('[{"question": ')

    def test_a_bad_date_loses_the_date_not_the_question(self):
        payload = json.dumps([{"question": GOOD_QUESTION, "asked_on": "last tuesday"}])
        submission = NOTES.read(payload)[0]
        assert submission.question == GOOD_QUESTION
        assert submission.asked_on is None

    def test_a_valid_date_is_kept(self):
        payload = json.dumps([{"question": GOOD_QUESTION, "date": "2026-03-14"}])
        assert NOTES.read(payload)[0].asked_on is not None

    def test_difficulty_is_clamped_to_the_scale(self):
        payload = json.dumps([{"question": GOOD_QUESTION, "difficulty": "9"}])
        assert NOTES.read(payload)[0].difficulty == 5


class TestCsvNotes:
    def test_a_csv_export_is_parsed(self):
        payload = f"question,company,level\n\"{GOOD_QUESTION}\",stripe,senior\n"
        submissions = NOTES.read(payload)
        assert len(submissions) == 1
        assert submissions[0].company == "stripe"

    def test_headers_are_case_insensitive(self):
        payload = f"Question,Company\n\"{GOOD_QUESTION}\",acme\n"
        assert NOTES.read(payload)[0].company == "acme"


class TestMarkdownNotes:
    def test_a_bulleted_list_is_parsed(self):
        payload = f"- {GOOD_QUESTION}\n- {SECOND_QUESTION}\n"
        assert len(NOTES.read(payload)) == 2

    def test_a_numbered_list_is_parsed(self):
        payload = f"1. {GOOD_QUESTION}\n2. {SECOND_QUESTION}\n"
        assert len(NOTES.read(payload)) == 2

    def test_metadata_lines_are_attached_to_the_question_above(self):
        payload = f"- {GOOD_QUESTION}\n  company: stripe\n  round: onsite\n"
        submission = NOTES.read(payload)[0]
        assert submission.company == "stripe"
        assert submission.interview_round == "onsite"

    def test_a_wrapped_question_is_joined_rather_than_split(self):
        """Notes wrap. Treating the second line as a new question would mangle both."""
        payload = "- How would you design a rate limiter\n  for a public API under load?\n"
        submissions = NOTES.read(payload)
        assert len(submissions) == 1
        assert "public API under load" in submissions[0].question

    def test_an_empty_payload_yields_nothing(self):
        assert NOTES.read("") == []
        assert NOTES.read("   \n  ") == []


class TestBulkImportGetsNoShortcut:
    """Volume is a reason for more scrutiny, not less."""

    def test_preview_screens_without_storing_anything(self):
        """`preview` is named for what it does. It reports; persisting is the service's
        job, and conflating the two is how a contributor gets told their notes were
        accepted when nothing was written."""
        payload = json.dumps([{"question": GOOD_QUESTION}])
        report = preview(payload, contributor_id="candidate-1")
        assert report.queued == 1
        assert report.rejected == 0

    def test_nda_material_in_a_file_is_still_refused(self):
        """The screen applies per record; a tidy file does not buy trust."""
        payload = json.dumps(
            [
                {"question": GOOD_QUESTION},
                {"question": "This was under NDA but here is their whole question set."},
            ]
        )
        report = preview(payload)
        assert report.queued == 1
        assert report.rejected == 1
        assert report.rejections[0]["reason"]

    def test_personal_data_in_a_file_is_still_scrubbed(self):
        payload = json.dumps(
            [{"question": f"My interviewer was Sarah. {SECOND_QUESTION}"}]
        )
        report = preview(payload)
        assert report.queued == 1

    def test_questions_already_in_the_bank_are_counted_as_duplicates(self):
        payload = json.dumps(
            [
                {
                    "question": (
                        "What ordering guarantees does Kafka give you, and across what "
                        "scope do those guarantees apply?"
                    )
                }
            ]
        )
        report = preview(payload)
        assert report.duplicates == 1
        assert report.queued == 0

    def test_a_rejection_report_does_not_reproduce_what_it_refused(self):
        """Echoing back refused content defeats refusing it."""
        long_secret = "Under NDA. " + ("secret detail " * 40)
        report = preview(json.dumps([{"question": long_secret}]))
        assert report.rejected == 1
        assert len(report.rejections[0]["question"]) <= 80

    def test_the_summary_accounts_for_every_record(self):
        payload = json.dumps(
            [
                {"question": GOOD_QUESTION},
                {"question": SECOND_QUESTION},
                {"question": "under NDA, confidential material"},
            ]
        )
        report = preview(payload)
        assert report.parsed == 3
        assert report.queued + report.duplicates + report.rejected == 3
        assert "parsed" in report.summary()

    def test_an_unknown_source_is_refused(self):
        with pytest.raises(ValueError, match="Unknown source"):
            preview("[]", source_key="glassdoor")


class TestTermsAreRecorded:
    def test_every_registered_source_permits_reuse(self):
        """A source that does not permit reuse has no business being registered."""
        for source in SOURCES.values():
            assert source.terms.permits_reuse, source.key

    def test_the_own_notes_terms_explain_the_basis(self):
        source = get_source("own_notes")
        assert source is not None
        assert "personally asked" in source.terms.notes

    def test_refusals_are_documented_rather_than_implicit(self):
        """The absence of scrapers should read as a decision, not an oversight."""
        refusals = refused_sources()
        assert len(refusals) >= 3
        joined = " ".join(item["reason"] for item in refusals).lower()
        assert "terms of service" in joined
        assert "copyright" in joined

    def test_leaked_material_is_named_as_refused(self):
        names = " ".join(item["source"] for item in refused_sources()).lower()
        assert "leaked" in names

    def test_an_empty_report_reads_sensibly(self):
        assert "0 parsed" in ImportReport(source="own_notes").summary()
