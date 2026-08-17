"""Source adapters for bulk question import (spec section 37).

An adapter turns some external format into candidate submissions and hands them to the
existing pipeline. It does not get its own path into the corpus: everything an adapter
produces goes through the same safety screen, deduplication and human review queue as a
question typed in by one person, because a thousand questions arriving at once is a
reason for more scrutiny, not less.

**Why there are no scraper adapters, and why that is not an omission.**

The obvious sources are the interview sites, and Gauntlet does not scrape them. Their
terms forbid it, the content is often someone else's copyrighted text, and a good share of
what is posted is material the poster was not free to share. The project already refuses
leaked assessments from an individual contributor; harvesting the same material in bulk
because it is on a web page would be the same refusal with the effort hidden behind an
adapter. The rule is the source's terms and the poster's right to share, not whether the
text is reachable over HTTP.

What is here instead is the adapter that is lawful and genuinely wanted: importing a
candidate's own notes. People keep a file of questions they were asked, and typing forty
of them one at a time through a web form is the reason they never get contributed.

Adding a lawful source means writing one class. The protocol is small on purpose, and the
per-source terms live with the adapter so they are auditable in one place.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import date
from io import StringIO
from typing import Protocol, runtime_checkable

import structlog

from gauntlet.ingestion.pipeline import Outcome, Submission, process

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SourceTerms:
    """What a source permits. Recorded so a licence question has one answer, not a guess."""

    name: str
    permits_reuse: bool
    requires_attribution: bool
    notes: str


@runtime_checkable
class Source(Protocol):
    """Anything that can produce submissions."""

    key: str
    terms: SourceTerms

    def read(self, payload: str) -> list[Submission]:
        """Parse a payload into submissions. Must not perform network access."""
        ...


@dataclass
class ImportReport:
    """What an import did, in enough detail to explain any single rejection."""

    source: str
    parsed: int = 0
    queued: int = 0
    duplicates: int = 0
    rejected: int = 0
    rejections: list[dict[str, str]] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return self.queued + self.duplicates

    def summary(self) -> str:
        return (
            f"{self.parsed} parsed, {self.queued} queued for review, "
            f"{self.duplicates} already known, {self.rejected} rejected"
        )


# --- The lawful adapter --------------------------------------------------------

OWN_NOTES_TERMS = SourceTerms(
    name="Candidate's own notes",
    permits_reuse=True,
    requires_attribution=False,
    notes=(
        "Questions the contributor was personally asked, in their own words. The "
        "contributor asserts they are free to share them, and the safety screen still "
        "refuses anything NDA-covered."
    ),
)

_FRONT_MATTER = re.compile(r"^\s*([A-Za-z_ ]{2,20}):\s*(.+?)\s*$")


class OwnNotesSource:
    """A candidate's own interview notes, as JSON, CSV or markdown.

    Three formats because people keep notes in whatever they already use, and a tool that
    only accepts one of them gets the notes retyped or not imported at all.
    """

    key = "own_notes"
    terms = OWN_NOTES_TERMS

    def read(self, payload: str) -> list[Submission]:
        text = payload.strip()
        if not text:
            return []
        if text.startswith(("[", "{")):
            return self._read_json(text)
        if self._looks_like_csv(text):
            return self._read_csv(text)
        return self._read_markdown(text)

    # -- format detection and parsing -------------------------------------
    @staticmethod
    def _looks_like_csv(text: str) -> bool:
        first = text.splitlines()[0].lower()
        return "," in first and "question" in first

    def _read_json(self, text: str) -> list[Submission]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON: {exc}") from exc
        rows = data if isinstance(data, list) else [data]
        return [self._from_mapping(row) for row in rows if isinstance(row, dict)]

    def _read_csv(self, text: str) -> list[Submission]:
        reader = csv.DictReader(StringIO(text))
        return [
            self._from_mapping({k.strip().lower(): v for k, v in row.items() if k})
            for row in reader
        ]

    def _read_markdown(self, text: str) -> list[Submission]:
        """Bullets or headings, with optional `key: value` lines beneath each.

        Deliberately forgiving. These are somebody's private notes, not a data
        interchange format, and rejecting the file because a line lacks a prefix would
        lose the notes rather than improve them.
        """
        submissions: list[Submission] = []
        current: dict[str, str] = {}

        def flush() -> None:
            if current.get("question"):
                submissions.append(self._from_mapping(dict(current)))
            current.clear()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("- ", "* ", "#")) or re.match(r"^\d+[.)]\s", line):
                flush()
                current["question"] = re.sub(r"^([-*#]+|\d+[.)])\s*", "", line).strip()
                continue
            match = _FRONT_MATTER.match(line)
            if match and current:
                current[match.group(1).strip().lower()] = match.group(2).strip()
            elif current:
                # A continuation line: part of the question, not metadata.
                current["question"] = f"{current['question']} {line}".strip()
        flush()
        return submissions

    @staticmethod
    def _from_mapping(row: dict[str, object]) -> Submission:
        def value(*names: str) -> str | None:
            for name in names:
                raw = row.get(name)
                if raw is not None and str(raw).strip():
                    return str(raw).strip()
            return None

        asked_on: date | None = None
        raw_date = value("asked_on", "date", "when")
        if raw_date:
            try:
                asked_on = date.fromisoformat(raw_date[:10])
            except ValueError:
                # A malformed date loses the date, not the question.
                asked_on = None

        difficulty: int | None = None
        raw_difficulty = value("difficulty", "level_of_difficulty")
        if raw_difficulty and raw_difficulty.isdigit():
            difficulty = max(1, min(5, int(raw_difficulty)))

        return Submission(
            question=value("question", "prompt", "text", "q") or "",
            company=value("company", "org", "employer"),
            role=value("role", "position", "title"),
            level=value("level", "seniority"),
            interview_round=value("round", "stage", "interview_round"),
            asked_on=asked_on,
            notes=value("notes", "comment", "context"),
            difficulty=difficulty,
        )


SOURCES: dict[str, Source] = {OwnNotesSource.key: OwnNotesSource()}


def get_source(key: str) -> Source | None:
    return SOURCES.get(key.strip().lower())


def parse(payload: str, *, source_key: str = "own_notes") -> list[Submission]:
    """Parse a payload into submissions, checking the source is one we may use."""
    source = get_source(source_key)
    if source is None:
        raise ValueError(f"Unknown source '{source_key}'. Known: {', '.join(SOURCES)}.")
    if not source.terms.permits_reuse:
        raise ValueError(
            f"Source '{source_key}' does not permit reuse: {source.terms.notes}"
        )
    return source.read(payload)


def preview(
    payload: str, *, source_key: str = "own_notes", contributor_id: str | None = None
) -> ImportReport:
    """Screen a payload and report what would happen, storing nothing.

    Deliberately named for what it does. This was originally called ``import_payload``
    and reported records as "queued for review" while writing nothing to the database,
    which is a worse failure than not having the feature: the contributor is told their
    notes were accepted and they are simply gone. Persisting lives in
    :func:`gauntlet.services.contributions.import_notes`, which needs a session and a
    candidate to attribute the rows to.
    """
    source = get_source(source_key)
    if source is None:
        raise ValueError(f"Unknown source '{source_key}'. Known: {', '.join(SOURCES)}.")
    if not source.terms.permits_reuse:
        raise ValueError(
            f"Source '{source_key}' does not permit reuse: {source.terms.notes}"
        )

    submissions = source.read(payload)
    report = ImportReport(source=source.key, parsed=len(submissions))

    for submission in submissions:
        submission.contributor_id = contributor_id
        result = process(submission)
        if result.outcome is Outcome.QUEUED:
            report.queued += 1
        elif result.outcome is Outcome.DUPLICATE:
            report.duplicates += 1
        else:
            report.rejected += 1
            report.rejections.append(
                {
                    # Truncated: a rejection report should not reproduce the content it
                    # just refused, especially when it was refused for containing
                    # personal data.
                    "question": submission.question[:80],
                    "reason": "; ".join(result.reasons)[:300],
                }
            )

    log.info(
        "ingestion.import",
        source=source.key,
        parsed=report.parsed,
        queued=report.queued,
        duplicates=report.duplicates,
        rejected=report.rejected,
    )
    return report


def refused_sources() -> list[dict[str, str]]:
    """Sources deliberately not implemented, and why.

    Written down rather than left implicit, so the absence reads as a decision instead of
    a gap somebody forgot to fill.
    """
    return [
        {
            "source": "Interview aggregator sites",
            "reason": (
                "Terms of service forbid automated collection, and the text is the "
                "poster's, not ours to republish."
            ),
        },
        {
            "source": "Paywalled question banks",
            "reason": "Reproducing paid content is straightforward copyright infringement.",
        },
        {
            "source": "Leaked assessments and answer keys",
            "reason": (
                "Refused from an individual contributor already. Collecting the same "
                "material in bulk does not change what it is."
            ),
        },
        {
            "source": "Company internal documents",
            "reason": "Confidential by definition, whoever supplies them.",
        },
    ]
