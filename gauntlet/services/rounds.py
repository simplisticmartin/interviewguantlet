"""Multi-round interview loops (spec section 4).

A real hiring process is not one interview. It is a phone screen, then two or three
technical rounds, then a hiring manager, and the rounds are not independent: what the
phone screen established is not re-established later, and a doubt raised in round two is
what round three goes after. Practising one isolated interview at a time misses the part
candidates actually find hardest, which is sustaining a level across a day.

A loop here is a **plan plus the sessions run against it**. There is no separate progress
table: the rounds that exist are the sessions carrying the loop id, so progress cannot
drift out of sync with reality the way a duplicated counter would.

**On simulated rejection.** Real loops drop people between rounds, and a practice tool
that never does is teaching the wrong shape. But Gauntlet is preparation, so a weak round
never locks the candidate out: it is reported as "in a real loop this is usually where you
would not advance", with the reason, and the next round stays available. The signal is
delivered; the practice is not confiscated. Anything else would be a tool that fails
someone based on a simulation and calls it a verdict.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

import structlog

from gauntlet.content.companies import find_company
from gauntlet.schemas import InterviewType

log = structlog.get_logger(__name__)


class RoundOutcome(StrEnum):
    """What a completed round says about progression."""

    STRONG = "strong"
    ADVANCE = "advance"
    BORDERLINE = "borderline"
    BELOW_BAR = "below_bar"
    NOT_RUN = "not_run"


# The thresholds a real debrief argues about. Named and in one place so they can be
# challenged, rather than scattered as magic numbers through the reporting code.
STRONG_SCORE = 0.80
ADVANCE_SCORE = 0.65
BORDERLINE_SCORE = 0.50


def outcome_for(score: float | None) -> RoundOutcome:
    if score is None:
        return RoundOutcome.NOT_RUN
    if score >= STRONG_SCORE:
        return RoundOutcome.STRONG
    if score >= ADVANCE_SCORE:
        return RoundOutcome.ADVANCE
    if score >= BORDERLINE_SCORE:
        return RoundOutcome.BORDERLINE
    return RoundOutcome.BELOW_BAR


@dataclass(frozen=True, slots=True)
class RoundSpec:
    """One round in the plan."""

    ordinal: int
    name: str
    interview_types: tuple[InterviewType, ...]
    minutes: int
    purpose: str
    # A screen that goes badly ends a real loop; a later round is one signal among many.
    is_gate: bool = False


@dataclass(frozen=True, slots=True)
class LoopPlan:
    """An ordered set of rounds for one company, role and level."""

    loop_id: str
    company_slug: str | None
    target_role: str
    target_level: str
    rounds: tuple[RoundSpec, ...]
    # Where the shape came from. The company catalogue holds estimates rather than
    # observed reports, and a loop built from an estimate has to say so wherever it
    # surfaces, or it becomes a claim about how a company actually interviews.
    evidence: str = "estimated"
    basis: str = "generic_loop"
    disclaimer: str = ""

    def round_at(self, ordinal: int) -> RoundSpec | None:
        return next((item for item in self.rounds if item.ordinal == ordinal), None)

    @property
    def total_minutes(self) -> int:
        return sum(item.minutes for item in self.rounds)


@dataclass(slots=True)
class RoundResult:
    """What a finished round contributed."""

    ordinal: int
    name: str
    session_id: str
    score: float | None
    outcome: RoundOutcome
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    concepts_covered: list[str] = field(default_factory=list)


def _screen(minutes: int, types: tuple[InterviewType, ...]) -> RoundSpec:
    return RoundSpec(
        ordinal=1,
        name="Phone screen",
        interview_types=types,
        minutes=minutes,
        purpose="Establish a baseline and check the resume holds up under one probe.",
        # The screen is the one genuine gate in most processes.
        is_gate=True,
    )


def build_plan(
    *,
    company_slug: str | None = None,
    target_role: str = "Software Engineer",
    target_level: str = "senior",
    loop_id: str | None = None,
) -> LoopPlan:
    """Build a loop from the company's known shape, falling back to a common one.

    The mix comes from the company catalogue, which stores estimates rather than claims,
    so a loop for a company we know little about is a generic loop and is described as
    one rather than being dressed up as insider knowledge.
    """
    company = find_company(company_slug) if company_slug else None

    distribution: dict[str, float] = {}
    evidence = "estimated"
    basis = "generic_loop"
    disclaimer = (
        "A generic loop shape. Gauntlet holds no interview reports for this role, so "
        "this is a common structure rather than anyone's actual process."
    )
    if company is not None:
        mix = company.interview_mix()
        # interview_mix() is typed dict[str, object]; the distribution is a nested dict
        # of type name to weight, so it is narrowed explicitly rather than assumed.
        raw = mix.get("distribution")
        distribution = (
            {str(key): float(value) for key, value in raw.items()}
            if isinstance(raw, dict)
            else {}
        )
        evidence = str(mix.get("evidence", "estimated"))
        basis = str(mix.get("basis", "generic_loop"))
        disclaimer = str(mix.get("disclaimer", disclaimer))

    valid = {item.value for item in InterviewType}
    ranked = [
        InterviewType(key)
        for key, _ in sorted(distribution.items(), key=lambda pair: pair[1], reverse=True)
        if key in valid
    ]

    # A sensible default loop when the catalogue has nothing useful. Deliberately the
    # shape most companies actually run, not an idealised one.
    if not ranked:
        ranked = [
            InterviewType.DSA,
            InterviewType.SYSTEM_DESIGN,
            InterviewType.JAVA,
            InterviewType.BEHAVIORAL,
        ]

    rounds: list[RoundSpec] = [_screen(30, (ranked[0],))]

    technical = [item for item in ranked[1:] if item is not InterviewType.BEHAVIORAL][:2]
    for offset, interview_type in enumerate(technical, start=2):
        rounds.append(
            RoundSpec(
                ordinal=offset,
                name=f"Technical round {offset - 1}",
                interview_types=(interview_type,),
                minutes=45,
                purpose=(
                    "Go deeper than the screen, and follow up on anything the screen "
                    "left uncertain."
                ),
            )
        )

    rounds.append(
        RoundSpec(
            ordinal=len(rounds) + 1,
            name="Hiring manager",
            interview_types=(InterviewType.HIRING_MANAGER, InterviewType.BEHAVIORAL),
            minutes=30,
            purpose="Scope, ownership, and how the candidate handles disagreement.",
        )
    )

    return LoopPlan(
        loop_id=loop_id or uuid.uuid4().hex,
        company_slug=company_slug,
        target_role=target_role,
        target_level=target_level,
        rounds=tuple(rounds),
        evidence=evidence,
        basis=basis,
        disclaimer=disclaimer,
    )


def carry_forward(results: list[RoundResult]) -> dict[str, object]:
    """What a later round should know about the earlier ones.

    This is what makes a loop different from four unrelated interviews. Concepts already
    demonstrated are not re-established, and concerns raised earlier become the things a
    later round goes after, which is exactly what a real panel does at its sync.
    """
    established: list[str] = []
    open_concerns: list[str] = []
    for result in results:
        if result.outcome in {RoundOutcome.STRONG, RoundOutcome.ADVANCE}:
            established.extend(result.concepts_covered)
        open_concerns.extend(result.concerns)

    return {
        "already_demonstrated": sorted(set(established)),
        "open_concerns": open_concerns[:8],
        "rounds_completed": len(results),
    }


@dataclass(slots=True)
class LoopStatus:
    """Where a loop stands, derived rather than stored."""

    plan: LoopPlan
    results: list[RoundResult]

    @property
    def completed(self) -> int:
        return len(self.results)

    @property
    def next_round(self) -> RoundSpec | None:
        return self.plan.round_at(self.completed + 1)

    @property
    def finished(self) -> bool:
        return self.completed >= len(self.plan.rounds)

    @property
    def average_score(self) -> float | None:
        scored = [r.score for r in self.results if r.score is not None]
        return sum(scored) / len(scored) if scored else None

    def would_have_advanced(self) -> bool:
        """Whether a real process would still have this candidate in it.

        Reported, never enforced: :attr:`next_round` stays available regardless. A
        practice tool that locks someone out on a simulated result has confused a
        rehearsal for a decision.
        """
        return all(
            result.outcome is not RoundOutcome.BELOW_BAR
            for result in self.results
            if (spec := self.plan.round_at(result.ordinal)) is not None and spec.is_gate
        )

    def progression_note(self) -> str:
        """Plain language on where this would have gone, and what to do next."""
        if not self.results:
            return "No rounds completed yet."

        failed_gates = [
            result
            for result in self.results
            if result.outcome is RoundOutcome.BELOW_BAR
            and (spec := self.plan.round_at(result.ordinal)) is not None
            and spec.is_gate
        ]
        if failed_gates:
            names = ", ".join(result.name for result in failed_gates)
            return (
                f"In a real loop this is usually where you would not advance ({names}). "
                "The remaining rounds are still open here, and running them is the point: "
                "practice is where that gets fixed."
            )

        weakest = min(
            (r for r in self.results if r.score is not None),
            key=lambda r: r.score or 0.0,
            default=None,
        )
        if self.finished:
            average = self.average_score
            if average is not None and average >= ADVANCE_SCORE:
                return (
                    f"Loop complete, averaging {average:.0%}. That is the shape of a "
                    "loop that gets to a hiring committee."
                )
            return (
                f"Loop complete, averaging {average:.0%}. " if average is not None else
                "Loop complete. "
            ) + (
                f"The round to work on is {weakest.name}." if weakest else ""
            )

        remaining = len(self.plan.rounds) - self.completed
        return (
            f"{self.completed} of {len(self.plan.rounds)} rounds done, {remaining} to go."
            + (f" Weakest so far: {weakest.name}." if weakest else "")
        )

    def debrief(self) -> dict[str, object]:
        """The combined packet, in the shape a hiring committee would read."""
        return {
            "loop_id": self.plan.loop_id,
            "company": self.plan.company_slug,
            "role": self.plan.target_role,
            "level": self.plan.target_level,
            "rounds_planned": len(self.plan.rounds),
            "loop_shape_evidence": self.plan.evidence,
            "loop_shape_basis": self.plan.basis,
            "rounds_completed": self.completed,
            "average_score": self.average_score,
            "would_have_advanced": self.would_have_advanced(),
            "progression": self.progression_note(),
            "rounds": [
                {
                    "ordinal": result.ordinal,
                    "name": result.name,
                    "session_id": result.session_id,
                    "score": result.score,
                    "outcome": result.outcome.value,
                    "strengths": result.strengths,
                    "concerns": result.concerns,
                }
                for result in self.results
            ],
            "carry_forward": carry_forward(self.results),
            "disclaimer": (
                "A simulated loop against an estimated interview shape. It is practice "
                "feedback, not a prediction of what any company would decide. "
                + self.plan.disclaimer
            ),
        }
