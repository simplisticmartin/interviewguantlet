"""Mastery model (spec section 23).

Deliberately *not* an average of scores. Each observation is weighted by how much it
actually tells us:

    mastery = sum(weight_i * adjusted_score_i) / sum(weight_i)

    weight_i        = recency x difficulty-informativeness x independence x hint-penalty
    adjusted_score_i= score, shifted toward the reference difficulty

and separately we track *our* confidence in that estimate, which grows with accumulated
evidence weight and shrinks when observations disagree with each other.

Everything here is a pure function over :class:`Evidence`. That is the point: the whole
model can be swapped for IRT or Bayesian knowledge tracing (the spec's stated end state)
by replacing this module, because the durable record in the database is raw evidence
rows, not the derived numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

# --- Tunables. Named constants, not magic numbers scattered through the code. ---
REFERENCE_DIFFICULTY = 3
DIFFICULTY_STEP = 0.05
RECENCY_HALF_LIFE_DAYS = 45.0
FOLLOWUP_INDEPENDENCE = 0.6
HINT_WEIGHT_PENALTY = 0.15
HINT_SCORE_PENALTY = 0.10
CONFIDENCE_SATURATION = 2.2
MIN_WEIGHT = 0.05

# Calibration thresholds (spec section 22).
MASTERY_HIGH = 0.7
MASTERY_LOW = 0.45
SELF_CONFIDENCE_HIGH = 0.7
SELF_CONFIDENCE_LOW = 0.45


class Calibration(StrEnum):
    """The knowledge-vs-confidence quadrant."""

    MASTERY = "mastery"
    CONFIDENCE_DEFICIT = "confidence_deficit"
    KNOWN_WEAKNESS = "known_weakness"
    MISCONCEPTION = "misconception"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One graded observation about one concept.

    ``observed_at`` defaults to *now* rather than a fixed date: recency weighting means
    a hardcoded default would make every timestamp-less observation quietly decay as the
    calendar advances, which is a bug that only shows up months later.
    """

    score: float
    difficulty: int = REFERENCE_DIFFICULTY
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    self_confidence: int | None = None
    hints_used: int = 0
    is_followup: bool = False
    judge_confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class MasteryState:
    mastery: float
    confidence: float
    evidence_count: int
    self_confidence: float | None
    calibration: Calibration

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "mastery": round(self.mastery, 4),
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "self_confidence": (
                round(self.self_confidence, 4) if self.self_confidence is not None else None
            ),
            "calibration": self.calibration.value,
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def recency_weight(
    observed_at: datetime, now: datetime, half_life_days: float | None = None
) -> float:
    """Exponential decay. Evidence from a year ago should not outvote last week's."""
    half_life = half_life_days if half_life_days is not None else RECENCY_HALF_LIFE_DAYS
    if half_life <= 0:
        return 1.0
    age_days = max((now - observed_at).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / half_life)


def difficulty_weight(difficulty: int) -> float:
    """Harder questions are more informative about the top of the range."""
    return 0.7 + 0.1 * (_clamp_difficulty(difficulty) - 1)


def _clamp_difficulty(difficulty: int) -> int:
    return max(1, min(5, difficulty))


def adjust_score_for_difficulty(score: float, difficulty: int) -> float:
    """Express a score as expected performance at the reference difficulty.

    Succeeding on a hard question is evidence of more mastery than the raw score says;
    failing an easy question is evidence of less.
    """
    level = _clamp_difficulty(difficulty)
    if level >= REFERENCE_DIFFICULTY:
        adjusted = score + DIFFICULTY_STEP * (level - REFERENCE_DIFFICULTY) * score
    else:
        adjusted = score - DIFFICULTY_STEP * (REFERENCE_DIFFICULTY - level) * (1.0 - score)
    return _clamp(adjusted)


def evidence_weight(item: Evidence, now: datetime) -> float:
    weight = recency_weight(item.observed_at, now) * difficulty_weight(item.difficulty)
    if item.is_followup:
        # A follow-up was scaffolded by the probe, so it is not independent evidence.
        weight *= FOLLOWUP_INDEPENDENCE
    weight *= max(0.2, 1.0 - HINT_WEIGHT_PENALTY * item.hints_used)
    # A judge that says "I am not sure" should not move the estimate as much.
    weight *= _clamp(item.judge_confidence, 0.2, 1.0)
    return max(weight, MIN_WEIGHT)


def normalise_self_confidence(raw: int | None) -> float | None:
    """1-5 Likert -> 0-1. 1 becomes 0.0, 5 becomes 1.0."""
    if raw is None:
        return None
    return _clamp((max(1, min(5, raw)) - 1) / 4.0)


def classify_calibration(mastery: float, self_confidence: float | None) -> Calibration:
    """The quadrant that drives study-plan priority (spec section 22)."""
    if self_confidence is None:
        return Calibration.UNKNOWN
    if mastery >= MASTERY_HIGH and self_confidence >= SELF_CONFIDENCE_HIGH:
        return Calibration.MASTERY
    if mastery >= MASTERY_HIGH and self_confidence < SELF_CONFIDENCE_LOW:
        return Calibration.CONFIDENCE_DEFICIT
    if mastery < MASTERY_LOW and self_confidence >= SELF_CONFIDENCE_HIGH:
        # The expensive quadrant: wrong and sure of it.
        return Calibration.MISCONCEPTION
    if mastery < MASTERY_LOW and self_confidence < SELF_CONFIDENCE_LOW:
        return Calibration.KNOWN_WEAKNESS
    return Calibration.UNKNOWN


def compute_mastery(evidence: list[Evidence], now: datetime | None = None) -> MasteryState:
    """Fold a concept's observations into a current belief."""
    if not evidence:
        return MasteryState(
            mastery=0.0,
            confidence=0.0,
            evidence_count=0,
            self_confidence=None,
            calibration=Calibration.UNKNOWN,
        )

    moment = now or datetime.now(UTC)
    weights: list[float] = []
    adjusted: list[float] = []

    for item in evidence:
        weight = evidence_weight(item, moment)
        score = adjust_score_for_difficulty(item.score, item.difficulty)
        score = _clamp(score - HINT_SCORE_PENALTY * item.hints_used)
        weights.append(weight)
        adjusted.append(score)

    total_weight = sum(weights)
    mastery = sum(w * s for w, s in zip(weights, adjusted, strict=True)) / total_weight

    # Our confidence in the estimate: more accumulated weight is better, disagreement
    # between observations is worse.
    saturation = 1.0 - math.exp(-total_weight / CONFIDENCE_SATURATION)
    mean = mastery
    variance = (
        sum(w * (s - mean) ** 2 for w, s in zip(weights, adjusted, strict=True)) / total_weight
    )
    agreement = 1.0 - _clamp(math.sqrt(variance) * 1.5)
    confidence = _clamp(saturation * (0.55 + 0.45 * agreement))

    self_scores = [
        normalise_self_confidence(item.self_confidence)
        for item in evidence
        if item.self_confidence is not None
    ]
    self_confidence = (
        sum(value for value in self_scores if value is not None) / len(self_scores)
        if self_scores
        else None
    )

    return MasteryState(
        mastery=_clamp(mastery),
        confidence=confidence,
        evidence_count=len(evidence),
        self_confidence=self_confidence,
        calibration=classify_calibration(_clamp(mastery), self_confidence),
    )


def next_review(
    mastery: float, previous_interval_days: int, now: datetime | None = None
) -> tuple[datetime, int]:
    """Spaced repetition schedule (spec section 30).

    Returns (due_at, next_interval_days). Weak concepts come back tomorrow; strong ones
    stretch out. The *question* is reworded elsewhere - this only decides timing.
    """
    moment = now or datetime.now(UTC)
    previous = max(1, previous_interval_days)

    if mastery >= 0.85:
        interval = min(int(previous * 3), 120)
    elif mastery >= 0.7:
        interval = min(int(previous * 2), 60)
    elif mastery >= 0.5:
        interval = min(max(int(previous * 1.4), 2), 21)
    else:
        interval = 1

    return moment + timedelta(days=interval), interval


def roll_up(
    child_states: dict[str, MasteryState], child_weights: dict[str, float] | None = None
) -> MasteryState:
    """Aggregate children into a parent concept reading.

    A parent is only as trustworthy as the evidence beneath it, so parent confidence is
    the evidence-weighted mean of child confidences, not an optimistic maximum.
    """
    if not child_states:
        return MasteryState(0.0, 0.0, 0, None, Calibration.UNKNOWN)

    weights = child_weights or {}
    total = 0.0
    mastery_sum = 0.0
    confidence_sum = 0.0
    evidence_count = 0
    self_values: list[float] = []

    for key, state in child_states.items():
        weight = weights.get(key, 1.0) * max(state.confidence, 0.1)
        total += weight
        mastery_sum += weight * state.mastery
        confidence_sum += weight * state.confidence
        evidence_count += state.evidence_count
        if state.self_confidence is not None:
            self_values.append(state.self_confidence)

    mastery = mastery_sum / total if total else 0.0
    confidence = confidence_sum / total if total else 0.0
    self_confidence = sum(self_values) / len(self_values) if self_values else None

    return MasteryState(
        mastery=_clamp(mastery),
        confidence=_clamp(confidence),
        evidence_count=evidence_count,
        self_confidence=self_confidence,
        calibration=classify_calibration(_clamp(mastery), self_confidence),
    )
