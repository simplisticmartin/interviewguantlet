"""In-memory skill graph used during a live interview.

Evidence lands here as answers are graded; readings are derived on demand. Ancestor
concepts (``java.concurrency`` above ``java.concurrency.volatile``) are rolled up rather
than stored, so the tree can never drift out of sync with the observations under it.

At the end of a session this is merged into the candidate's persistent skill state,
which additionally carries evidence from every previous interview.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from gauntlet.content.taxonomy import ancestors_of, deeper_concepts, display_name, get_concept
from gauntlet.schemas import SkillReading
from gauntlet.skills.mastery import Calibration, Evidence, MasteryState, compute_mastery, roll_up


@dataclass
class SkillGraph:
    """Concept -> observations, with derived readings."""

    evidence: dict[str, list[Evidence]] = field(default_factory=lambda: defaultdict(list))
    misconception_keys: set[str] = field(default_factory=set)

    # --- Mutation --------------------------------------------------------

    def record(self, concept_keys: list[str], item: Evidence) -> None:
        """Attach one observation to every concept the question exercised."""
        for key in concept_keys:
            self.evidence[key].append(item)

    def flag_misconception(self, concept_key: str | None) -> None:
        if concept_key:
            self.misconception_keys.add(concept_key)

    # --- Reading ---------------------------------------------------------

    def observed_keys(self) -> list[str]:
        return sorted(self.evidence)

    def state_for(self, concept_key: str, now: datetime | None = None) -> MasteryState:
        direct = self.evidence.get(concept_key)
        if direct:
            return compute_mastery(direct, now)

        # No direct evidence: roll up anything observed beneath this concept.
        prefix = f"{concept_key}."
        children = {
            key: compute_mastery(items, now)
            for key, items in self.evidence.items()
            if key.startswith(prefix)
        }
        if children:
            return roll_up(children)
        return compute_mastery([], now)

    def reading(self, concept_key: str, now: datetime | None = None) -> SkillReading:
        state = self.state_for(concept_key, now)
        return SkillReading(
            concept_key=concept_key,
            display_name=display_name(concept_key),
            mastery=state.mastery,
            confidence=state.confidence,
            evidence_count=state.evidence_count,
            self_confidence=state.self_confidence,
            is_misconception=(
                concept_key in self.misconception_keys
                or state.calibration is Calibration.MISCONCEPTION
            ),
        )

    def all_readings(self, now: datetime | None = None) -> list[SkillReading]:
        """Readings for observed concepts plus every ancestor they imply."""
        keys: set[str] = set(self.evidence)
        for key in list(keys):
            keys.update(ancestors_of(key))
        return sorted(
            (self.reading(key, now) for key in keys),
            key=lambda reading: reading.mastery,
        )

    def leaf_readings(self, now: datetime | None = None) -> list[SkillReading]:
        """Only directly-observed concepts - what the interview actually measured."""
        return sorted(
            (self.reading(key, now) for key in self.evidence),
            key=lambda reading: reading.mastery,
        )

    def strongest(self, limit: int = 5, now: datetime | None = None) -> list[SkillReading]:
        readings = [r for r in self.leaf_readings(now) if r.evidence_count]
        return sorted(readings, key=lambda r: r.mastery, reverse=True)[:limit]

    def weakest(self, limit: int = 5, now: datetime | None = None) -> list[SkillReading]:
        readings = [r for r in self.leaf_readings(now) if r.evidence_count]
        return sorted(readings, key=lambda r: r.mastery)[:limit]

    # --- Interview steering ---------------------------------------------

    def evidence_count(self, concept_key: str) -> int:
        return len(self.evidence.get(concept_key, []))

    def is_saturated(self, concept_key: str, threshold: float = 0.85, min_count: int = 2) -> bool:
        """True when more questions here would waste interview time."""
        items = self.evidence.get(concept_key)
        if not items or len(items) < min_count:
            return False
        state = compute_mastery(items)
        return state.mastery >= threshold and state.confidence >= 0.5

    def next_deeper(self, concept_key: str) -> str | None:
        """The next unexplored concept below this one."""
        for candidate in deeper_concepts(concept_key):
            if candidate not in self.evidence:
                return candidate
        return None

    def to_dict(
        self, now: datetime | None = None
    ) -> dict[str, dict[str, float | int | str | None]]:
        moment = now or datetime.now(UTC)
        return {key: self.state_for(key, moment).as_dict() for key in sorted(self.evidence)}

    @classmethod
    def from_persisted(cls, rows: list[tuple[str, Evidence]]) -> SkillGraph:
        """Rebuild from stored evidence rows (used for cross-interview history)."""
        graph = cls()
        for concept_key, item in rows:
            graph.evidence[concept_key].append(item)
        return graph


def category_scores(readings: list[SkillReading]) -> dict[str, int]:
    """Roll leaf readings up to top-level domains for the report's headline bars."""
    buckets: dict[str, list[SkillReading]] = defaultdict(list)
    for reading in readings:
        if not reading.evidence_count:
            continue
        root = reading.concept_key.split(".")[0]
        buckets[root].append(reading)

    scores: dict[str, int] = {}
    for root, group in buckets.items():
        concept = get_concept(root)
        label = concept.display_name if concept else root.replace("_", " ").title()
        weighted = sum(r.mastery * max(r.confidence, 0.2) for r in group)
        total = sum(max(r.confidence, 0.2) for r in group)
        scores[label] = round(100 * (weighted / total)) if total else 0
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
