"""Spoken interview handling (spec section 44).

Built above the speech layer. No recognition or synthesis vendor is wired up, so voice
interviews cannot run end to end yet; see :func:`readiness` for exactly what exists.
"""

from gauntlet.voice.transcript import (
    LONG_PAUSE_SECONDS,
    PRESERVED_HEDGES,
    SpokenAnswer,
    Utterance,
    assemble,
    is_thinking_aloud,
    normalise,
    readiness,
    turn_has_ended,
)

__all__ = [
    "LONG_PAUSE_SECONDS",
    "PRESERVED_HEDGES",
    "SpokenAnswer",
    "Utterance",
    "assemble",
    "is_thinking_aloud",
    "normalise",
    "readiness",
    "turn_has_ended",
]
