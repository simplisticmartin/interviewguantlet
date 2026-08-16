"""Offline question variant generation (spec section 39).

The test that matters most is `test_every_variant_still_asks_its_source_question`. It runs
every generated variant back through the deduplication module and requires it to still be
detected as a duplicate of the question it came from. That is a real check on the
invariant rather than a restatement of it: a template that drifted far enough to change
what is being asked would leave the rubric grading something nobody was asked, and this
catches it mechanically.
"""

from __future__ import annotations

import pytest

from gauntlet.content.questions import QUESTIONS
from gauntlet.content.variants import (
    TEMPLATES,
    Framing,
    applicable_templates,
    build_variant,
    generate_variants,
    pick_variant,
    variant_count,
)
from gauntlet.ingestion.dedup import QuestionCandidate, compare
from gauntlet.schemas import InterviewType

BY_SLUG = {seed.slug: seed for seed in QUESTIONS}
HASHMAP = BY_SLUG["java-hashmap-internals"]


class TestMeaningIsPreserved:
    @pytest.mark.parametrize("seed", QUESTIONS, ids=lambda s: s.slug)
    def test_every_variant_still_asks_its_source_question(self, seed):
        """The invariant, checked against the deduplicator rather than by eye."""
        source = QuestionCandidate(
            id=seed.slug, text=seed.question, concept_keys=tuple(seed.concept_keys)
        )
        for variant in generate_variants(seed):
            candidate = QuestionCandidate(
                id=variant.slug, text=variant.text, concept_keys=variant.concept_keys
            )
            score = compare(candidate, source)
            assert score.is_duplicate, (
                f"{variant.slug} drifted from its source: {score.explain()}"
            )

    @pytest.mark.parametrize("seed", QUESTIONS, ids=lambda s: s.slug)
    def test_the_original_question_survives_verbatim(self, seed):
        """Templates wrap; they never rewrite."""
        for variant in generate_variants(seed):
            assert seed.question in variant.text

    def test_the_rubric_is_carried_across_unchanged(self):
        """A variant is graded by the rubric of the question it reframes."""
        for variant in generate_variants(HASHMAP):
            assert variant.rubric_key == HASHMAP.rubric_key
            assert variant.concept_keys == tuple(HASHMAP.concept_keys)

    def test_no_template_asks_a_question_of_its_own(self):
        """A second question in the prompt would be ungraded by the rubric."""
        for template in TEMPLATES:
            for fragment in (*template.lead_ins, *template.follow_ons):
                assert "?" not in fragment, fragment


class TestFraming:
    def test_the_direct_framing_is_the_question_itself(self):
        direct = build_variant(HASHMAP, TEMPLATES[0])
        assert direct.framing is Framing.DIRECT
        assert direct.text == HASHMAP.question

    def test_a_scenario_framing_adds_context_before_the_question(self):
        variants = {v.framing: v for v in generate_variants(HASHMAP)}
        scenario = variants[Framing.SCENARIO]
        assert scenario.text.endswith(HASHMAP.question)
        assert len(scenario.text) > len(HASHMAP.question)

    def test_generation_is_deterministic(self):
        """Same question, same framing, same text: interviews stay reproducible."""
        first = {v.framing: v.text for v in generate_variants(HASHMAP)}
        second = {v.framing: v.text for v in generate_variants(HASHMAP)}
        assert first == second

    def test_different_questions_get_different_lead_ins(self):
        """A stable hash, not a constant: otherwise every question reads identically."""
        seeds = [s for s in QUESTIONS if s.interview_type in {InterviewType.JAVA}][:8]
        leads = set()
        for seed in seeds:
            for variant in generate_variants(seed):
                if variant.framing is Framing.SCENARIO:
                    leads.add(variant.text.replace(seed.question, "").strip())
        assert len(leads) > 1

    def test_behavioural_questions_get_no_technical_framings(self):
        """A production incident is not a sensible wrapper for a behavioural question."""
        behavioural = [
            s for s in QUESTIONS if s.interview_type is InterviewType.BEHAVIORAL
        ]
        if not behavioural:
            pytest.skip("no behavioural questions in the corpus")
        for seed in behavioural:
            framings = {v.framing for v in generate_variants(seed)}
            assert framings == {Framing.DIRECT}

    def test_a_code_review_framing_only_applies_to_code_questions(self):
        for seed in QUESTIONS:
            framings = {v.framing for v in generate_variants(seed)}
            if Framing.CODE_REVIEW in framings:
                assert seed.expects_code, seed.slug


class TestDifficulty:
    def test_a_harder_framing_raises_difficulty(self):
        variants = {v.framing: v for v in generate_variants(HASHMAP)}
        assert variants[Framing.INCIDENT].difficulty == HASHMAP.difficulty + 1

    def test_difficulty_never_leaves_the_scale(self):
        """The rubric and the mastery model both assume 1 to 5."""
        for seed in QUESTIONS:
            for variant in generate_variants(seed):
                assert 1 <= variant.difficulty <= 5

    def test_a_five_stays_a_five(self):
        hardest = [s for s in QUESTIONS if s.difficulty == 5]
        if not hardest:
            pytest.skip("no difficulty 5 questions in the corpus")
        for seed in hardest:
            assert all(v.difficulty == 5 for v in generate_variants(seed))


class TestProvenance:
    def test_a_variant_is_marked_generated(self):
        """Nothing generated may ever be presented as observed."""
        for variant in generate_variants(HASHMAP):
            assert variant.provenance["question_origin"] == "generated"
            assert variant.provenance["generated_from"] == HASHMAP.slug

    def test_the_note_disclaims_the_framing(self):
        for variant in generate_variants(HASHMAP):
            assert "not a question any company is known" in variant.provenance["note"].lower()

    def test_slugs_are_unique_and_traceable_to_the_source(self):
        slugs = [v.slug for v in generate_variants(HASHMAP)]
        assert len(slugs) == len(set(slugs))
        assert all(slug.startswith(HASHMAP.slug) for slug in slugs)


class TestSelection:
    def test_it_skips_framings_already_seen(self):
        first = pick_variant(HASHMAP)
        assert first is not None
        second = pick_variant(HASHMAP, seen_slugs=frozenset({first.slug}))
        assert second is not None
        assert second.slug != first.slug

    def test_it_returns_none_rather_than_repeating(self):
        """Running out is a real condition; the caller should widen, not repeat."""
        every = frozenset(v.slug for v in generate_variants(HASHMAP))
        assert pick_variant(HASHMAP, seen_slugs=every) is None

    def test_the_corpus_gains_meaningful_depth(self):
        """The reason this exists: a fourth interview should not be a rerun."""
        total = sum(variant_count(seed) for seed in QUESTIONS)
        assert total > len(QUESTIONS) * 2

    def test_every_question_has_at_least_the_direct_framing(self):
        for seed in QUESTIONS:
            assert applicable_templates(seed), seed.slug
            assert variant_count(seed) >= 1


class TestTheInvariantHasHeadroom:
    """A variant that only just clears the duplicate threshold is a latent failure.

    The first version of the incident and handover templates cleared it by 0.013, which
    means a slightly shorter question added to the corpus, or any retuning of the dedup
    weights, would have broken the invariant. This pins the margin so that creeping
    verbosity in a template is caught when it is added rather than much later.
    """

    MINIMUM_MARGIN = 0.05

    def test_no_variant_sits_on_the_threshold(self):
        from gauntlet.ingestion.dedup import DUPLICATE_THRESHOLD

        worst: tuple[float, str] = (1.0, "")
        for seed in QUESTIONS:
            source = QuestionCandidate(
                id=seed.slug, text=seed.question, concept_keys=tuple(seed.concept_keys)
            )
            for variant in generate_variants(seed):
                candidate = QuestionCandidate(
                    id=variant.slug, text=variant.text, concept_keys=variant.concept_keys
                )
                score = compare(candidate, source).combined
                if score < worst[0]:
                    worst = (score, variant.slug)

        assert worst[0] >= DUPLICATE_THRESHOLD + self.MINIMUM_MARGIN, (
            f"{worst[1]} clears the duplicate threshold by only "
            f"{worst[0] - DUPLICATE_THRESHOLD:.3f}; the wrapper is too long for the "
            "question it wraps"
        )

    def test_a_wrapper_never_dwarfs_its_question(self):
        """Reads badly for the candidate, and dilutes the question for the matcher."""
        for seed in QUESTIONS:
            for variant in generate_variants(seed):
                wrapper = len(variant.text) - len(seed.question)
                assert wrapper <= len(seed.question), (
                    f"{variant.slug}: {wrapper} characters of framing around a "
                    f"{len(seed.question)} character question"
                )


class TestTheDirectFramingIsNotANewQuestion:
    """Regression: the direct framing is the source question, not a variant of it.

    The first version offered `slug~direct` as unseen whenever only `slug` was in the
    seen set, so the interviewer asked the same question twice. Every test in this file
    passed; it only showed up in the behaviour suite, because the defect was in what
    "seen" means rather than in how variants are built.
    """

    def test_a_seen_question_does_not_come_back_as_its_direct_framing(self):
        picked = pick_variant(HASHMAP, seen_slugs=frozenset({HASHMAP.slug}))
        assert picked is not None
        assert picked.framing is not Framing.DIRECT
        assert picked.text != HASHMAP.question

    def test_seen_may_be_given_as_prompt_text(self):
        """Callers track what has been asked by text as well as by slug."""
        picked = pick_variant(HASHMAP, seen_slugs=frozenset({HASHMAP.question}))
        assert picked is not None
        assert picked.text != HASHMAP.question

    def test_a_variant_already_asked_by_text_is_not_offered_again(self):
        first = pick_variant(HASHMAP, seen_slugs=frozenset({HASHMAP.slug}))
        assert first is not None
        second = pick_variant(
            HASHMAP, seen_slugs=frozenset({HASHMAP.slug, first.text})
        )
        assert second is not None
        assert second.text != first.text

    def test_the_direct_framing_is_still_offered_for_an_unseen_question(self):
        """Plain is best when nothing has been asked yet."""
        picked = pick_variant(HASHMAP)
        assert picked is not None
        assert picked.framing is Framing.DIRECT

    def test_repeated_picking_never_yields_the_same_text_twice(self):
        seen: set[str] = set()
        texts: list[str] = []
        while (picked := pick_variant(HASHMAP, seen_slugs=frozenset(seen))) is not None:
            texts.append(picked.text)
            seen.add(picked.slug)
        assert len(texts) == len(set(texts))
        assert len(texts) == variant_count(HASHMAP)
