"""Multi-round interview loops (spec section 4).

The tests that matter most are in `TestPracticeIsNeverConfiscated`. A practice tool that
simulates a rejection and then locks the candidate out has confused a rehearsal for a
decision, and that is a product failure rather than a bug, so it is pinned here.
"""

from __future__ import annotations

import pytest

from gauntlet.schemas import InterviewType
from gauntlet.services.rounds import (
    ADVANCE_SCORE,
    LoopStatus,
    RoundOutcome,
    RoundResult,
    build_plan,
    carry_forward,
    outcome_for,
)


def result(
    ordinal: int,
    score: float | None,
    *,
    name: str = "Round",
    concerns: list[str] | None = None,
    concepts: list[str] | None = None,
) -> RoundResult:
    return RoundResult(
        ordinal=ordinal,
        name=name,
        session_id=f"session-{ordinal}",
        score=score,
        outcome=outcome_for(score),
        concerns=concerns or [],
        concepts_covered=concepts or [],
    )


class TestPlanning:
    def test_a_loop_starts_with_a_screen_and_ends_with_a_manager(self):
        plan = build_plan(company_slug="google", target_level="senior")
        assert plan.rounds[0].name == "Phone screen"
        assert plan.rounds[-1].name == "Hiring manager"

    def test_the_screen_is_the_gate(self):
        """Most real processes gate on the screen and treat later rounds as signals."""
        plan = build_plan(company_slug="google")
        assert plan.rounds[0].is_gate
        assert not any(item.is_gate for item in plan.rounds[1:])

    def test_ordinals_are_contiguous(self):
        plan = build_plan(company_slug="stripe")
        assert [item.ordinal for item in plan.rounds] == list(range(1, len(plan.rounds) + 1))

    def test_an_unknown_company_still_gets_a_sensible_loop(self):
        """No catalogue entry means a generic loop, not a broken one."""
        plan = build_plan(company_slug="a-company-that-does-not-exist")
        assert len(plan.rounds) >= 3
        assert plan.total_minutes > 60

    def test_no_company_at_all_is_fine(self):
        assert len(build_plan().rounds) >= 3

    def test_the_manager_round_is_not_a_technical_round(self):
        plan = build_plan(company_slug="google")
        assert InterviewType.HIRING_MANAGER in plan.rounds[-1].interview_types

    def test_behavioural_does_not_take_a_technical_slot(self):
        """Otherwise the loop has no technical depth, which is not what a loop is."""
        plan = build_plan(company_slug="google")
        middle = plan.rounds[1:-1]
        for item in middle:
            assert InterviewType.BEHAVIORAL not in item.interview_types

    def test_each_loop_gets_its_own_id(self):
        assert build_plan().loop_id != build_plan().loop_id

    def test_an_id_can_be_supplied_for_resuming(self):
        assert build_plan(loop_id="fixed").loop_id == "fixed"


class TestOutcomes:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.95, RoundOutcome.STRONG),
            (0.80, RoundOutcome.STRONG),
            (0.70, RoundOutcome.ADVANCE),
            (0.55, RoundOutcome.BORDERLINE),
            (0.20, RoundOutcome.BELOW_BAR),
            (None, RoundOutcome.NOT_RUN),
        ],
    )
    def test_scores_map_to_outcomes(self, score, expected):
        assert outcome_for(score) is expected

    def test_an_unrun_round_is_not_a_failed_round(self):
        """Absent evidence is not negative evidence."""
        assert outcome_for(None) is not RoundOutcome.BELOW_BAR


class TestCarryForward:
    def test_a_later_round_knows_what_was_established(self):
        """This is what makes a loop different from four unrelated interviews."""
        carried = carry_forward(
            [result(1, 0.85, concepts=["kafka.ordering", "java.concurrency"])]
        )
        assert "kafka.ordering" in carried["already_demonstrated"]

    def test_a_weak_round_does_not_establish_anything(self):
        carried = carry_forward([result(1, 0.30, concepts=["kafka.ordering"])])
        assert carried["already_demonstrated"] == []

    def test_concerns_become_what_the_next_round_probes(self):
        carried = carry_forward([result(1, 0.55, concerns=["hand-waved on idempotency"])])
        assert "hand-waved on idempotency" in carried["open_concerns"]

    def test_concerns_are_bounded(self):
        """A panel sync raises a handful of doubts, not forty."""
        many = result(1, 0.5, concerns=[f"concern {i}" for i in range(30)])
        assert len(carry_forward([many])["open_concerns"]) <= 8


class TestProgress:
    def test_the_next_round_follows_the_completed_ones(self):
        plan = build_plan(company_slug="google")
        status = LoopStatus(plan=plan, results=[result(1, 0.8)])
        assert status.next_round is not None
        assert status.next_round.ordinal == 2

    def test_a_finished_loop_has_no_next_round(self):
        plan = build_plan(company_slug="google")
        results = [result(i, 0.8) for i in range(1, len(plan.rounds) + 1)]
        status = LoopStatus(plan=plan, results=results)
        assert status.finished
        assert status.next_round is None

    def test_the_average_ignores_rounds_that_did_not_run(self):
        status = LoopStatus(
            plan=build_plan(), results=[result(1, 0.80), result(2, None)]
        )
        assert status.average_score == pytest.approx(0.80)

    def test_an_empty_loop_has_no_average_rather_than_zero(self):
        """Zero would read as "scored badly" instead of "has not been attempted"."""
        assert LoopStatus(plan=build_plan(), results=[]).average_score is None


class TestPracticeIsNeverConfiscated:
    """A simulated rejection reports; it never locks the candidate out."""

    def test_a_failed_gate_is_reported(self):
        plan = build_plan(company_slug="google")
        status = LoopStatus(
            plan=plan, results=[result(1, 0.20, name=plan.rounds[0].name)]
        )
        assert status.would_have_advanced() is False
        assert "would not advance" in status.progression_note()

    def test_a_failed_gate_still_leaves_the_next_round_available(self):
        """The point of the whole design: the signal lands, the practice continues."""
        plan = build_plan(company_slug="google")
        status = LoopStatus(
            plan=plan, results=[result(1, 0.20, name=plan.rounds[0].name)]
        )
        assert status.next_round is not None
        assert not status.finished

    def test_the_note_tells_the_candidate_to_keep_going(self):
        plan = build_plan(company_slug="google")
        status = LoopStatus(plan=plan, results=[result(1, 0.10, name=plan.rounds[0].name)])
        note = status.progression_note()
        assert "still open" in note

    def test_a_weak_non_gate_round_does_not_end_the_loop(self):
        """One bad technical round is a signal, not a rejection."""
        plan = build_plan(company_slug="google")
        status = LoopStatus(
            plan=plan, results=[result(1, 0.85), result(2, 0.20)]
        )
        assert status.would_have_advanced() is True

    def test_the_debrief_never_presents_itself_as_a_real_decision(self):
        plan = build_plan(company_slug="google")
        status = LoopStatus(plan=plan, results=[result(1, 0.9)])
        debrief = status.debrief()
        assert "not a prediction" in debrief["disclaimer"]

    def test_a_strong_loop_is_described_as_such(self):
        plan = build_plan(company_slug="google")
        results = [result(i, 0.85) for i in range(1, len(plan.rounds) + 1)]
        note = LoopStatus(plan=plan, results=results).progression_note()
        assert "hiring committee" in note

    def test_the_debrief_carries_every_round(self):
        plan = build_plan(company_slug="google")
        results = [result(i, 0.7) for i in range(1, 3)]
        debrief = LoopStatus(plan=plan, results=results).debrief()
        assert len(debrief["rounds"]) == 2
        assert debrief["average_score"] == pytest.approx(0.7)
        assert debrief["average_score"] >= ADVANCE_SCORE


class TestTheLoopShapeIsNotPresentedAsFact:
    """The company catalogue holds estimates, and a loop built from one must say so."""

    def test_a_known_company_carries_its_evidence_level(self):
        plan = build_plan(company_slug="google")
        assert plan.evidence == "estimated"
        assert plan.basis.startswith("archetype:")

    def test_the_company_disclaimer_reaches_the_debrief(self):
        plan = build_plan(company_slug="google")
        debrief = LoopStatus(plan=plan, results=[result(1, 0.8)]).debrief()
        assert "simulation" in debrief["disclaimer"].lower()
        assert debrief["loop_shape_evidence"] == "estimated"

    def test_an_unknown_company_says_the_shape_is_generic(self):
        plan = build_plan(company_slug="not-a-real-company")
        assert plan.basis == "generic_loop"
        assert "generic" in plan.disclaimer.lower()

    def test_no_plan_claims_observed_evidence(self):
        """Nothing in the corpus supports claiming a loop shape was observed."""
        for slug in (None, "google", "stripe", "unknown-co"):
            assert build_plan(company_slug=slug).evidence != "observed"
