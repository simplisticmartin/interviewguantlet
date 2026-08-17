"""The contribution pipeline (spec sections 37 and 38).

Turns a raw contributed question into either a reviewable corpus candidate or a clear
refusal, running the spec's stages in the only order that is safe:

    safety screen -> normalise -> concept tag -> classify -> dedup -> provenance -> queue

**Safety runs first, before anything else touches the text.** Screening after tagging or
deduplication would mean an NDA-covered submission had already been embedded, indexed and
compared against the corpus before anyone decided it should be refused.

**Nothing here publishes.** Every accepted submission lands in a review queue with status
``pending``. That is the spec's requirement and it is also the only defensible design: an
automated pipeline deciding what a public question bank contains, with no human in the
loop, is how a corpus ends up with somebody's name in it.

The output carries full provenance, so an approved question is distinguishable forever
from the authored corpus. A contributed question is ``question_origin="user_submitted"``
and is never presented as though a company asked it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

import structlog

from gauntlet.content.questions import QUESTIONS
from gauntlet.content.taxonomy import CONCEPTS, get_concept
from gauntlet.ingestion.dedup import QuestionCandidate, compare, find_duplicates_of
from gauntlet.ingestion.safety import SafetyReport, Verdict, screen
from gauntlet.schemas import InterviewType

log = structlog.get_logger(__name__)

MIN_QUESTION_CHARS = 15
MAX_QUESTION_CHARS = 2000

# Similarity high enough to be worth a reviewer's attention, but not high enough to merge
# on automatically. The gap between this and DUPLICATE_THRESHOLD is where human judgement
# beats a number: "is this a real variant or the same question reworded?" is exactly the
# call a person should make, and it widens when no embedding provider is configured and
# only the lexical signal is available.
NEAR_DUPLICATE_FLOOR = 0.55


class Outcome(StrEnum):
    """What the pipeline decided about a submission."""

    QUEUED = "queued"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


@dataclass
class Submission:
    """What a contributor sends."""

    question: str
    company: str | None = None
    role: str | None = None
    level: str | None = None
    interview_round: str | None = None
    asked_on: date | None = None
    notes: str | None = None
    difficulty: int | None = None
    contributor_id: str | None = None


@dataclass
class PipelineResult:
    outcome: Outcome
    safety: SafetyReport
    question: str = ""
    concept_keys: list[str] = field(default_factory=list)
    interview_type: InterviewType | None = None
    difficulty: int = 3
    duplicate_of: str | None = None
    duplicate_similarity: float | None = None
    near_duplicates: list[tuple[str, float]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.outcome is Outcome.QUEUED


def tag_concepts(text: str, limit: int = 5) -> list[str]:
    """Map free text to taxonomy concepts by surface form.

    Deliberately conservative: it only matches concepts whose display name or alias
    actually appears. Guessing a concept would put a question in front of candidates
    under a label it does not deserve, which corrupts the skill measurement downstream.
    """
    lowered = re.sub(r"[\s_\-]+", " ", text.lower())
    scored: list[tuple[int, str]] = []

    for concept in CONCEPTS:
        surfaces = [concept.display_name, *concept.aliases]
        hits = 0
        for surface in surfaces:
            if not surface:
                continue
            normalised = re.sub(r"[\s_\-]+", " ", surface.lower())
            pattern = (
                r"\b"
                + r"[\s_\-]*".join(re.escape(part) for part in normalised.split())
                + r"\b"
            )
            if re.search(pattern, lowered):
                hits += 1
        if hits:
            # Prefer the most specific concept: deeper keys are more precise.
            scored.append((hits * 10 + concept.key.count("."), concept.key))

    scored.sort(reverse=True)
    return [key for _, key in scored[:limit]]


def infer_interview_type(concept_keys: list[str]) -> InterviewType | None:
    """Interview type from the tagged concepts, by majority."""
    counts: dict[InterviewType, int] = {}
    for key in concept_keys:
        concept = get_concept(key)
        if concept is None:
            continue
        counts[concept.interview_type] = counts.get(concept.interview_type, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda pair: pair[1])[0]


def estimate_difficulty(concept_keys: list[str], stated: int | None) -> int:
    """Contributor's estimate if given, otherwise the concept's own floor."""
    if stated is not None and 1 <= stated <= 5:
        return stated
    floors = [
        concept.difficulty_floor
        for key in concept_keys
        if (concept := get_concept(key)) is not None
    ]
    return max(floors) if floors else 3


def corpus_candidates() -> list[QuestionCandidate]:
    return [
        QuestionCandidate(
            id=seed.slug,
            text=seed.question,
            concept_keys=tuple(seed.concept_keys),
            topics=tuple(seed.topics),
        )
        for seed in QUESTIONS
    ]


def process(
    submission: Submission,
    *,
    corpus: list[QuestionCandidate] | None = None,
) -> PipelineResult:
    """Run one submission through the pipeline."""
    text = (submission.question or "").strip()

    # --- Stage 0: shape ----------------------------------------------------
    if len(text) < MIN_QUESTION_CHARS:
        return PipelineResult(
            outcome=Outcome.REJECTED,
            safety=SafetyReport(verdict=Verdict.BLOCK, text=text),
            reasons=[f"Too short to be a question (minimum {MIN_QUESTION_CHARS} characters)."],
        )
    if len(text) > MAX_QUESTION_CHARS:
        return PipelineResult(
            outcome=Outcome.REJECTED,
            safety=SafetyReport(verdict=Verdict.BLOCK, text=text),
            reasons=[f"Too long (maximum {MAX_QUESTION_CHARS} characters)."],
        )

    # --- Stage 1: safety, before anything else touches it ------------------
    combined = text if not submission.notes else f"{text}\n\n{submission.notes}"
    safety = screen(combined)
    if safety.verdict is Verdict.BLOCK:
        log.info(
            "ingestion.blocked",
            kinds=[kind.value for kind in safety.kinds()],
            contributor=submission.contributor_id,
        )
        return PipelineResult(
            outcome=Outcome.REJECTED, safety=safety, reasons=safety.reasons
        )

    cleaned = screen(text).text

    # --- Stage 2 and 3: tag and classify -----------------------------------
    concept_keys = tag_concepts(cleaned)
    interview_type = infer_interview_type(concept_keys)
    difficulty = estimate_difficulty(concept_keys, submission.difficulty)

    reasons = list(safety.reasons)
    if not concept_keys:
        # Not a refusal: an untagged question is still useful, a human can tag it.
        reasons.append("No taxonomy concept matched; needs manual tagging.")
    elif len(concept_keys) == 1 and "." not in concept_keys[0]:
        # Only a top-level branch matched, which is too coarse to slot a question with.
        reasons.append(
            f"Only matched the broad '{concept_keys[0]}' area; needs a specific concept."
        )

    # --- Stage 4: deduplicate ----------------------------------------------
    candidate = QuestionCandidate(
        id="__submission__", text=cleaned, concept_keys=tuple(concept_keys)
    )
    against = corpus if corpus is not None else corpus_candidates()
    matches = find_duplicates_of(candidate, against)

    # Anything close but under the merge threshold is handed to the reviewer rather than
    # decided automatically.
    near = [
        (existing.id, round(score.combined, 4))
        for existing in against
        if (score := compare(candidate, existing)).combined >= NEAR_DUPLICATE_FLOOR
        and not score.is_duplicate
    ]
    near.sort(key=lambda pair: pair[1], reverse=True)

    if matches:
        existing, score = matches[0]
        log.info("ingestion.duplicate", matched=existing.id, similarity=score.combined)
        return PipelineResult(
            outcome=Outcome.DUPLICATE,
            safety=safety,
            question=cleaned,
            concept_keys=concept_keys,
            interview_type=interview_type,
            difficulty=difficulty,
            duplicate_of=existing.id,
            duplicate_similarity=score.combined,
            reasons=[
                f"Already in the bank as '{existing.id}' ({score.explain()}). "
                "Recorded as another occurrence rather than a new question."
            ],
        )

    # --- Stage 5: provenance -----------------------------------------------
    provenance = {
        "question_origin": "user_submitted",
        "source_type": "user_contribution",
        "copyright_status": "contributor_asserted_original",
        "company": submission.company,
        "role": submission.role,
        "level": submission.level,
        "round": submission.interview_round,
        "asked_on": submission.asked_on.isoformat() if submission.asked_on else None,
        "received_at": datetime.now(UTC).isoformat(),
        "contributor_id": submission.contributor_id,
        "safety_findings": [finding.kind.value for finding in safety.findings],
        "attribution_note": (
            "Contributed by a user. Not verified, and not evidence that this company "
            "asks this question."
        ),
    }

    if near:
        top_id, top_score = near[0]
        reasons.append(
            f"Similar to existing question '{top_id}' ({top_score:.2f}), below the "
            "automatic merge threshold. Confirm it is genuinely a different question."
        )

    # --- Stage 6: queue, never publish -------------------------------------
    log.info(
        "ingestion.queued",
        concepts=len(concept_keys),
        interview_type=interview_type.value if interview_type else None,
        needs_review=safety.verdict is Verdict.REVIEW,
        near_duplicates=len(near),
    )
    return PipelineResult(
        outcome=Outcome.QUEUED,
        safety=safety,
        question=cleaned,
        concept_keys=concept_keys,
        interview_type=interview_type,
        difficulty=difficulty,
        near_duplicates=near[:5],
        reasons=reasons,
        provenance=provenance,
    )
