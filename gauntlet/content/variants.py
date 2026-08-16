"""Offline question variant generation (spec section 39).

A corpus of 75 questions is thin once a candidate sits their fourth interview. They start
seeing the same wording, which stops measuring knowledge and starts measuring memory of
this tool.

Variants re-frame an existing question without an LLM, so they work on the offline path
and cost nothing. The generation is deterministic: the same question and framing always
produce the same text, which keeps interviews reproducible and makes the output testable.

**The invariant: framing changes, meaning does not.**

This is the whole risk. Every question is graded against a rubric keyed to what it asks.
A template that quietly changes the question ("walk me through a HashMap put" becoming
"compare HashMap and TreeMap") leaves the rubric grading something that was never asked,
and the candidate is marked down for correctly answering the question in front of them.
So templates only ever wrap: they add a situation before the question and a constraint
after it, and never rewrite the question itself.

That invariant is verified rather than asserted. Every generated variant is run back
through :mod:`gauntlet.ingestion.dedup` and must still be detected as a duplicate of its
source. A template that drifted far enough to change the question fails the test suite.

**Provenance.** A variant is ``question_origin="generated"``, always. It is never
presented as something a company asked, because no company asked it: Gauntlet wrote it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from gauntlet.content.questions import QuestionSeed
from gauntlet.schemas import InterviewType

# Framings that would change what is being tested are not here. Every one of these adds
# context around the question while leaving the question itself intact.


class Framing(StrEnum):
    """How a question is put, not what it asks."""

    DIRECT = "direct"
    SCENARIO = "scenario"
    INCIDENT = "incident"
    CODE_REVIEW = "code_review"
    HANDOVER = "handover"
    CONSTRAINED = "constrained"


@dataclass(frozen=True, slots=True)
class FramingTemplate:
    """A situation to set before the question, and a constraint to add after it.

    Both are optional. Neither may contain a question: the moment a template asks
    something of its own, the rubric no longer covers the whole prompt.
    """

    framing: Framing
    lead_ins: tuple[str, ...] = ()
    follow_ons: tuple[str, ...] = ()
    difficulty_delta: int = 0
    # Empty means it applies anywhere. Some framings only make sense for some rounds:
    # a production incident is not a sensible wrapper for a behavioural question.
    interview_types: frozenset[InterviewType] = frozenset()
    requires_code: bool | None = None


_TECHNICAL = frozenset(
    {
        InterviewType.DSA,
        InterviewType.JAVA,
        InterviewType.SPRING,
        InterviewType.DATABASE,
        InterviewType.DISTRIBUTED,
        InterviewType.SYSTEM_DESIGN,
        InterviewType.CLOUD,
        InterviewType.FRONTEND,
        InterviewType.AI_ENGINEERING,
    }
)

TEMPLATES: tuple[FramingTemplate, ...] = (
    FramingTemplate(framing=Framing.DIRECT),
    FramingTemplate(
        framing=Framing.SCENARIO,
        lead_ins=(
            "You have just joined a team that owns this area.",
            "Imagine you are the engineer on call for this service.",
            "A teammate asks you this in a design discussion.",
        ),
        interview_types=_TECHNICAL,
    ),
    FramingTemplate(
        framing=Framing.INCIDENT,
        # Kept short on purpose. A long preamble plus a trailing instruction buries the
        # question for the candidate, and dilutes it enough that the deduplicator stops
        # recognising the variant as the same question. Both are the same failure: the
        # wrapper has taken over from what is being asked.
        lead_ins=(
            "This is happening in production right now.",
            "An alert has just fired on this.",
        ),
        # An incident frame adds time pressure, which is genuinely harder than answering
        # the same thing at a whiteboard.
        difficulty_delta=1,
        interview_types=_TECHNICAL,
    ),
    FramingTemplate(
        framing=Framing.CODE_REVIEW,
        lead_ins=("You are reviewing a colleague's pull request that touches this.",),
        interview_types=_TECHNICAL,
        requires_code=True,
    ),
    FramingTemplate(
        framing=Framing.HANDOVER,
        # One wrapper, not two, for the same reason as the incident framing: the shortest
        # questions in the corpus are around 65 characters, and a lead-in plus a trailer
        # is more wrapper than question.
        lead_ins=(
            "Explain this to an engineer new to the system.",
            "You are handing this over to someone unfamiliar with it.",
        ),
        interview_types=_TECHNICAL,
    ),
    FramingTemplate(
        framing=Framing.CONSTRAINED,
        follow_ons=(
            "Now assume ten times the traffic.",
            "You cannot add new infrastructure.",
            "There is no budget for new tooling.",
        ),
        difficulty_delta=1,
        interview_types=_TECHNICAL,
    ),
)


@dataclass(frozen=True, slots=True)
class QuestionVariant:
    """A reframed question, carrying the source it must still be graded against."""

    text: str
    framing: Framing
    source_slug: str
    concept_keys: tuple[str, ...]
    interview_type: InterviewType
    difficulty: int
    # Unchanged from the source, deliberately: the variant asks the same thing, so it is
    # graded by the same rubric. A variant that needed a different rubric would be a
    # different question.
    rubric_key: str | None
    follow_ups: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    expects_code: bool = False
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return f"{self.source_slug}~{self.framing.value}"


def _stable_choice[T](options: tuple[T, ...], *parts: str) -> T:
    """Pick deterministically from a hash of the inputs.

    Deterministic rather than random so an interview can be reproduced from its
    checkpoint, and so a test asserting on generated text is not flaky.
    """
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return options[int.from_bytes(digest[:4], "big") % len(options)]


def applicable_templates(seed: QuestionSeed) -> tuple[FramingTemplate, ...]:
    """Templates that make sense for this question."""
    usable = []
    for template in TEMPLATES:
        if template.interview_types and seed.interview_type not in template.interview_types:
            continue
        if template.requires_code is not None and template.requires_code != seed.expects_code:
            continue
        usable.append(template)
    return tuple(usable)


def build_variant(seed: QuestionSeed, template: FramingTemplate) -> QuestionVariant:
    """Wrap one question in one framing."""
    parts: list[str] = []

    if template.lead_ins:
        parts.append(_stable_choice(template.lead_ins, seed.slug, template.framing.value))

    parts.append(seed.question)

    if template.follow_ons:
        parts.append(
            _stable_choice(template.follow_ons, seed.slug, template.framing.value, "after")
        )

    # Clamped: a delta must not push a question outside the 1 to 5 scale the rubric and
    # the mastery model both assume.
    difficulty = max(1, min(5, seed.difficulty + template.difficulty_delta))

    return QuestionVariant(
        text=" ".join(parts),
        framing=template.framing,
        source_slug=seed.slug,
        concept_keys=tuple(seed.concept_keys),
        interview_type=seed.interview_type,
        difficulty=difficulty,
        rubric_key=seed.rubric_key,
        follow_ups=tuple(seed.follow_ups),
        topics=tuple(seed.topics),
        expects_code=seed.expects_code,
        provenance={
            "question_origin": "generated",
            "source_type": "gauntlet_authored",
            "generated_from": seed.slug,
            "framing": template.framing.value,
            "method": "offline_template",
            "note": (
                "Reframed from an authored question. Not a question any company is "
                "known to have asked in this form."
            ),
        },
    )


def generate_variants(
    seed: QuestionSeed, *, exclude_framings: frozenset[str] = frozenset()
) -> list[QuestionVariant]:
    """Every usable framing of one question, most conservative first."""
    return [
        build_variant(seed, template)
        for template in applicable_templates(seed)
        if template.framing.value not in exclude_framings
    ]


def pick_variant(
    seed: QuestionSeed, *, seen_slugs: frozenset[str] = frozenset()
) -> QuestionVariant | None:
    """The best unseen framing of a question, or ``None`` when they are all used up.

    ``seen`` may hold slugs or prompt text: callers track "already used" both ways, and a
    variant matching either is not new. Checking only slugs let a variant through whose
    text the candidate had already been asked.

    The direct framing is the source question verbatim, so once the source has been asked
    the direct framing is by definition already seen. Missing that was a real defect: the
    interviewer offered the same question a second time because ``slug~direct`` was not
    in the set even though ``slug`` was.

    ``None`` rather than repeating: running out of ways to ask something is a real
    condition, and the caller should widen the concept rather than be handed a question
    the candidate has already answered.
    """
    source_seen = seed.slug in seen_slugs or seed.question in seen_slugs

    for variant in generate_variants(seed):
        if variant.framing is Framing.DIRECT and source_seen:
            continue
        if variant.slug in seen_slugs or variant.text in seen_slugs:
            continue
        return variant
    return None


def variant_count(seed: QuestionSeed) -> int:
    """How many distinct ways this question can be put."""
    return len(applicable_templates(seed))


def as_seed(variant: QuestionVariant, source: QuestionSeed) -> QuestionSeed:
    """Present a variant as a ``QuestionSeed``.

    Lets a variant travel through retrieval, the interviewer agent and grading using the
    paths that already exist, instead of a parallel set that would each need to learn
    about variants. The rubric key and concept keys are the source's, which is what keeps
    grading correct.
    """
    return QuestionSeed(
        slug=variant.slug,
        question=variant.text,
        interview_type=variant.interview_type,
        concept_keys=variant.concept_keys,
        difficulty=variant.difficulty,
        rubric_key=variant.rubric_key,
        follow_ups=variant.follow_ups,
        topics=variant.topics,
        level=source.level,
        expects_code=variant.expects_code,
        asks_confidence=source.asks_confidence,
        role_family=source.role_family,
        based_on_patterns=source.based_on_patterns,
    )
