"""Spoken answer handling (spec section 44).

`TestHedgingIsNeverRemoved` is the class that matters. Stripping disfluency is a fairness
fix; stripping hedging would be a correctness bug, because "I think it is probably B" and
"it is B" are different answers and the confidence calibration model reads the difference.
"""

from __future__ import annotations

import pytest

from gauntlet.voice import (
    PRESERVED_HEDGES,
    Utterance,
    assemble,
    is_thinking_aloud,
    normalise,
    readiness,
    turn_has_ended,
)


class TestDisfluencyRemoval:
    """Speech is disfluent. Grading a transcript as prose penalises speaking normally."""

    @pytest.mark.parametrize("filler", ["um", "uh", "erm", "er", "ah"])
    def test_common_fillers_are_removed(self, filler: str):
        cleaned, removed = normalise(f"So {filler} the answer is a hash map.")
        assert filler not in cleaned.lower().split()
        assert removed >= 1

    def test_stutters_are_collapsed(self):
        cleaned, _ = normalise("The the map resizes when it it gets full.")
        assert "the the" not in cleaned.lower()
        assert "it it" not in cleaned.lower()

    def test_false_starts_are_removed(self):
        cleaned, _ = normalise("It uses conc- concurrent hash maps under load.")
        assert "conc-" not in cleaned
        assert "concurrent hash maps" in cleaned

    def test_discourse_fillers_are_removed_only_as_filler(self):
        cleaned, _ = normalise("So, like, the bucket becomes a tree.")
        assert "like," not in cleaned.lower()

    def test_meaningful_like_survives(self):
        """A comparison is content, not filler."""
        cleaned, _ = normalise("It works like a hash map with buckets.")
        assert "like a hash map" in cleaned

    def test_the_technical_content_is_untouched(self):
        spoken = "Um, so, ConcurrentHashMap uh uses lock striping, I think."
        cleaned, _ = normalise(spoken)
        assert "ConcurrentHashMap" in cleaned
        assert "lock striping" in cleaned

    def test_empty_speech_is_handled(self):
        assert normalise("") == ("", 0)
        assert normalise("   ") == ("", 0)

    def test_clean_speech_is_left_alone(self):
        text = "The load factor triggers a resize at seventy five percent."
        cleaned, removed = normalise(text)
        assert cleaned == text
        assert removed == 0


class TestHedgingIsNeverRemoved:
    """Hedging is content. Removing it inflates the answer and corrupts calibration."""

    @pytest.mark.parametrize("hedge", PRESERVED_HEDGES)
    def test_every_hedge_survives(self, hedge: str):
        cleaned, _ = normalise(f"Um, {hedge} it resizes at seventy five percent.")
        assert hedge in cleaned.lower()

    def test_an_uncertain_answer_stays_uncertain(self):
        cleaned, _ = normalise("Um, I think it is probably a red-black tree, not sure.")
        lowered = cleaned.lower()
        assert "i think" in lowered
        assert "probably" in lowered
        assert "not sure" in lowered

    def test_normalisation_does_not_turn_a_guess_into_a_claim(self):
        hedged, _ = normalise("Uh, maybe it is O(n log n).")
        assert "maybe" in hedged.lower()


class TestAssembly:
    def _utterances(self) -> list[Utterance]:
        return [
            Utterance("Um, so the map", 1.0, 3.0),
            Utterance("resizes when it gets full.", 3.5, 6.0),
        ]

    def test_the_original_is_kept_beside_the_cleaned_version(self):
        answer = assemble(self._utterances())
        assert "Um" in answer.raw_text
        assert "um" not in answer.text.lower().split()

    def test_interim_results_are_discarded(self):
        """Grading a partial transcript scores an answer that was not finished."""
        answer = assemble(
            [
                Utterance("The map res", 1.0, 2.0, is_final=False),
                Utterance("The map resizes at the load factor.", 1.0, 4.0),
            ]
        )
        assert "res." not in answer.text
        assert "resizes at the load factor" in answer.text

    def test_out_of_order_chunks_are_sorted(self):
        answer = assemble(
            [Utterance("second part.", 5.0, 7.0), Utterance("First part", 1.0, 3.0)]
        )
        assert answer.text.startswith("First part")

    def test_silence_produces_an_explicit_result(self):
        answer = assemble([])
        assert answer.text == ""
        assert "No speech" in answer.observations[0]

    def test_duration_and_rate_are_computed(self):
        answer = assemble(self._utterances())
        assert answer.duration_seconds == pytest.approx(5.0)
        assert answer.words_per_minute > 0


class TestTimingIsObservedNotScored:
    def test_a_long_pause_before_answering_is_noted(self):
        answer = assemble(
            [Utterance("The answer is a tree.", 12.0, 14.0)], question_ended_at=0.0
        )
        assert any("before" in note for note in answer.observations)

    def test_a_short_pause_is_not_remarked_on(self):
        answer = assemble(
            [Utterance("The answer is a tree.", 1.0, 3.0)], question_ended_at=0.0
        )
        assert not any("Paused" in note for note in answer.observations)

    def test_a_long_mid_answer_silence_suggests_a_prompt(self):
        answer = assemble(
            [
                Utterance("So the first thing", 0.0, 2.0),
                Utterance("is the load factor.", 15.0, 17.0),
            ]
        )
        assert any("silence mid-answer" in note for note in answer.observations)

    def test_a_slow_answer_is_described_neutrally(self):
        """Slow is not wrong, and the wording must not imply it is."""
        answer = assemble([Utterance("The load factor.", 0.0, 20.0)])
        notes = " ".join(answer.observations).lower()
        assert "wrong" not in notes
        assert "poor" not in notes

    def test_low_recognition_confidence_warns_against_fine_grading(self):
        """A bad transcript is not a bad answer and must never be graded as one."""
        answer = assemble([Utterance("It uses lock striping.", 0.0, 3.0, confidence=0.3)])
        assert answer.low_confidence
        assert any("low confidence" in note.lower() for note in answer.observations)

    def test_disfluency_is_reported_as_coaching_not_as_a_penalty(self):
        answer = assemble([Utterance("Um uh er um so uh the map um resizes uh", 0.0, 5.0)])
        notes = " ".join(answer.observations).lower()
        if "disfluent" in notes:
            assert "not scored" in notes
            assert "penalty" not in notes


class TestTurnTaking:
    def test_a_finished_sentence_after_silence_ends_the_turn(self):
        assert turn_has_ended("That is my answer.", silence_seconds=3.0)

    def test_a_brief_pause_does_not_end_the_turn(self):
        assert not turn_has_ended("The map resizes", silence_seconds=0.5)

    @pytest.mark.parametrize(
        "text",
        [
            "Let me think",
            "So if the bucket is full and",
            "Okay so",
            "Hmm",
            "The tradeoff is between throughput and",
        ],
    )
    def test_thinking_aloud_is_recognised(self, text: str):
        assert is_thinking_aloud(text)

    def test_a_thinking_candidate_gets_longer_before_being_cut_off(self):
        """Cutting someone off for thinking punishes the behaviour the format asks for."""
        assert not turn_has_ended("Let me think", silence_seconds=3.0)
        assert turn_has_ended("That is my final answer.", silence_seconds=3.0)

    def test_silence_with_nothing_said_never_ends_a_turn(self):
        assert not turn_has_ended("", silence_seconds=30.0)

    def test_a_trailing_conjunction_means_unfinished(self):
        assert is_thinking_aloud("It resizes because")


class TestHonestyAboutWhatIsBuilt:
    def test_readiness_admits_no_vendor_is_wired(self):
        state = readiness()
        assert state["speech_to_text_vendor"] is False
        assert state["text_to_speech_vendor"] is False
        assert "cannot be run end to end" in str(state["note"])

    def test_the_parts_that_do_exist_are_claimed(self):
        state = readiness()
        assert state["transcript_normalisation"] is True
        assert state["turn_taking"] is True
