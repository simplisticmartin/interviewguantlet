"""Spoken answer handling (spec section 44).

**No speech recognition or synthesis vendor is wired up.** This module sits above that
line: it takes whatever a speech-to-text service produced and turns it into something the
existing graders can score fairly. Choosing and wiring an STT vendor is the remaining
work, and it is called out as such rather than stubbed with a fake that would make the
feature look finished.

The part that is built here is the part that is actually hard, and it is a fairness
problem rather than a plumbing one.

**Spoken answers are not written answers.** A transcript of a good spoken answer contains
"um", restarts, self-corrections and repetition. Every one of those is normal speech, and
none of them says anything about whether the candidate knows the material. Feed that
transcript to a grader calibrated on written answers and it marks the candidate down for
speaking like a person. That penalty does not fall evenly: it falls hardest on people
speaking a second language, on anyone nervous, and on people who think out loud, which is
the behaviour interviewers explicitly ask for.

So the transcript is normalised before grading and the original is kept beside it. What
is removed is disfluency; what is never removed is content, hedging or uncertainty, since
"I think it is probably B" and "it is B" are genuinely different answers and the second
is not an improvement on the first. Hedging is signal for the confidence calibration
model, so erasing it would corrupt a measurement the product depends on.

**Timing is kept, not discarded.** A long pause before an answer, or a long silence in the
middle, is real interview signal that written answers cannot carry. It is recorded as an
observation for the interviewer, never folded into the score, because slow is not wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

import structlog

log = structlog.get_logger(__name__)

# Words that carry no content in speech. Deliberately short: an aggressive list starts
# removing things that mean something. "Like" is here only in its filler position, which
# the pattern below handles, because "works like a hash map" is content.
FILLERS = (
    "um",
    "uh",
    "erm",
    "hmm",
    "mm",
    "ah",
    "er",
)

_FILLER_PATTERN = re.compile(
    r"\b(?:" + "|".join(FILLERS) + r")\b[,.]?\s*", re.IGNORECASE
)
# "like" and "you know" only when they sit between commas or start a clause, which is
# where they are filler rather than meaning.
_DISCOURSE_FILLER = re.compile(
    r"(?:^|(?<=[,.]\s))\s*(?:like|you know|i mean|sort of|kind of)\s*,\s*", re.IGNORECASE
)
# A stutter or restart: "the the", "a a", "I I".
_REPEATED_WORD = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)
# A false start abandoned mid-word then restarted: "conc- concurrent".
_FALSE_START = re.compile(r"\b\w{1,6}-\s+(?=\w)")
_WHITESPACE = re.compile(r"\s+")

# Never removed. These change what the answer claims, and the confidence calibration
# model reads them directly.
PRESERVED_HEDGES = (
    "i think",
    "i believe",
    "probably",
    "maybe",
    "might",
    "not sure",
    "i guess",
    "possibly",
    "if i remember",
)

# Long enough to mean something, short enough to happen in a normal answer.
LONG_PAUSE_SECONDS = 4.0
VERY_LONG_PAUSE_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class Utterance:
    """One chunk of recognised speech with its timing."""

    text: str
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    # Most STT services report a confidence. Low confidence means the transcript may be
    # wrong, which is a very different thing from the candidate being wrong.
    confidence: float = 1.0
    is_final: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


@dataclass(slots=True)
class SpokenAnswer:
    """A spoken answer, in both the form a person said it and the form a grader reads."""

    raw_text: str
    text: str
    words: int = 0
    fillers_removed: int = 0
    duration_seconds: float = 0.0
    time_to_first_word: float = 0.0
    longest_pause_seconds: float = 0.0
    low_confidence: bool = False
    observations: list[str] = field(default_factory=list)

    @property
    def words_per_minute(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.words / (self.duration_seconds / 60.0)

    @property
    def disfluency_rate(self) -> float:
        """Fillers per hundred words. An observation, never a penalty."""
        return (self.fillers_removed / self.words * 100) if self.words else 0.0


def normalise(text: str) -> tuple[str, int]:
    """Strip disfluency, keep meaning. Returns the cleaned text and how much was removed.

    Order matters. False starts are removed before repeated words, or "conc- concurrent"
    leaves a fragment that the repeat pattern then cannot match.
    """
    if not text.strip():
        return "", 0

    original_words = len(text.split())

    cleaned = _FALSE_START.sub("", text)
    cleaned = _FILLER_PATTERN.sub(" ", cleaned)
    cleaned = _DISCOURSE_FILLER.sub(" ", cleaned)
    cleaned = _REPEATED_WORD.sub(r"\1", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()

    # Tidy punctuation left stranded by removals.
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = re.sub(r"^[,.\s]+", "", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)

    removed = max(0, original_words - len(cleaned.split()))
    return cleaned, removed


def assemble(
    utterances: list[Utterance], *, question_ended_at: float = 0.0
) -> SpokenAnswer:
    """Turn recognised speech into a gradable answer plus timing observations.

    Interim results are dropped. Grading a partial transcript scores an answer the
    candidate had not finished giving.
    """
    final = [item for item in utterances if item.is_final and item.text.strip()]
    if not final:
        return SpokenAnswer(raw_text="", text="", observations=["No speech was recognised."])

    ordered = sorted(final, key=lambda item: item.start_seconds)
    raw = " ".join(item.text.strip() for item in ordered)
    cleaned, removed = normalise(raw)

    start = ordered[0].start_seconds
    end = max(item.end_seconds for item in ordered)

    longest_pause = 0.0
    for earlier, later in pairwise(ordered):
        longest_pause = max(longest_pause, later.start_seconds - earlier.end_seconds)

    lowest_confidence = min(item.confidence for item in ordered)

    answer = SpokenAnswer(
        raw_text=raw,
        text=cleaned,
        words=len(cleaned.split()),
        fillers_removed=removed,
        duration_seconds=max(0.0, end - start),
        time_to_first_word=max(0.0, start - question_ended_at),
        longest_pause_seconds=longest_pause,
        low_confidence=lowest_confidence < 0.6,
    )
    answer.observations = _observe(answer)

    log.info(
        "voice.answer",
        words=answer.words,
        fillers_removed=answer.fillers_removed,
        duration_s=round(answer.duration_seconds, 1),
        low_confidence=answer.low_confidence,
    )
    return answer


def _observe(answer: SpokenAnswer) -> list[str]:
    """Timing notes for the interviewer.

    These are observations, not scores. Every one of them is phrased so that a human
    reading the debrief understands it as context rather than a deduction, because
    thinking before answering is a good habit and a tool that punishes it is teaching
    people to bluff faster.
    """
    notes: list[str] = []

    if answer.time_to_first_word >= VERY_LONG_PAUSE_SECONDS:
        notes.append(
            f"Took {answer.time_to_first_word:.0f}s before starting. Worth noticing "
            "whether that was thinking or being stuck; both are fine, but they lead to "
            "different follow-ups."
        )
    elif answer.time_to_first_word >= LONG_PAUSE_SECONDS:
        notes.append(f"Paused {answer.time_to_first_word:.0f}s before answering.")

    if answer.longest_pause_seconds >= VERY_LONG_PAUSE_SECONDS:
        notes.append(
            f"A {answer.longest_pause_seconds:.0f}s silence mid-answer, which usually "
            "means the thread was lost. A prompt here is what a real interviewer offers."
        )

    if answer.words and answer.words_per_minute > 200:
        notes.append("Spoke quickly, which can read as rushing under pressure.")
    elif answer.words and 0 < answer.words_per_minute < 90:
        notes.append("Spoke slowly and deliberately.")

    if answer.low_confidence:
        notes.append(
            "Parts of the transcript were recognised with low confidence, so wording may "
            "not be exactly what was said. Do not grade fine detail from it."
        )

    if answer.disfluency_rate > 15:
        notes.append(
            "Noticeably disfluent delivery. Removed before grading and not scored; "
            "mentioned only because delivery is coachable if the candidate asks."
        )
    return notes


def is_thinking_aloud(text: str) -> bool:
    """Whether the speaker is reasoning rather than finishing.

    Interviewers ask candidates to think out loud, so an answer must not be cut off just
    because the speaker paused. This is what a silence timer should consult before
    deciding a turn has ended.
    """
    lowered = text.lower().strip()
    markers = (
        "let me think",
        "so if",
        "hmm",
        "one second",
        "give me a moment",
        "i'm thinking",
        "let's see",
        "okay so",
        "wait",
        "actually",
    )
    if any(lowered.endswith(marker) for marker in markers):
        return True
    if any(marker in lowered[-40:] for marker in markers):
        return True
    # A trailing conjunction means the sentence is not finished.
    return bool(re.search(r"\b(and|but|so|because|which|then|or)\s*$", lowered))


def turn_has_ended(
    text: str, silence_seconds: float, *, patience_seconds: float = 2.5
) -> bool:
    """Whether to treat the candidate as finished speaking.

    Patience is doubled while someone is audibly mid-thought. Cutting a candidate off
    because they paused to think is the single most irritating failure a voice
    interviewer can have, and it punishes exactly the behaviour the format asks for.
    """
    if not text.strip():
        return False
    threshold = patience_seconds * 2 if is_thinking_aloud(text) else patience_seconds
    return silence_seconds >= threshold


def readiness() -> dict[str, object]:
    """What is and is not built, so nothing here looks more finished than it is."""
    return {
        "transcript_normalisation": True,
        "turn_taking": True,
        "timing_observations": True,
        "speech_to_text_vendor": False,
        "text_to_speech_vendor": False,
        "audio_transport": False,
        "note": (
            "Voice handling is built above the speech layer. No recognition or synthesis "
            "vendor is wired up, so voice interviews cannot be run end to end yet."
        ),
    }
