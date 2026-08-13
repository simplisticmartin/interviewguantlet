"""Agent behaviour tests (spec section 46).

These assert on what the interviewer *does*, not on wording:

  perfect answer            -> difficulty increases / descends deeper
  confidently wrong answer  -> misconception flagged AND a probe follows
  honest "I don't know"     -> scored zero, but never flagged as a misconception

They run entirely on the deterministic provider, so a failure here means the interview
logic changed, not that a model had an off day.
"""

from __future__ import annotations

from gauntlet.agents.classifier import ResponseClassifierAgent
from gauntlet.agents.interviewer import InterviewerAgent, QuestionTarget
from gauntlet.agents.router import AdaptiveRouterAgent, RoutingContext
from gauntlet.evaluation.engine import EvaluationEngine
from gauntlet.schemas import (
    AdaptiveDirection,
    AnswerPayload,
    InterviewType,
    QuestionSpec,
    ResponseClass,
)
from gauntlet.skills.graph import SkillGraph
from gauntlet.skills.mastery import Evidence
from tests.conftest import CONFIDENTLY_WRONG_KAFKA, STRONG_HASHMAP_ANSWER

HASHMAP_QUESTION = QuestionSpec(
    prompt_text="Walk me through a HashMap put that collides.",
    interview_type=InterviewType.JAVA,
    agent_key="java",
    concept_keys=["java.collections.hashmap"],
    difficulty=3,
    rubric_key="java.collections.hashmap",
)

CHM_QUESTION = QuestionSpec(
    prompt_text="How does ConcurrentHashMap differ from a synchronized HashMap?",
    interview_type=InterviewType.JAVA,
    agent_key="java",
    concept_keys=["java.concurrency.concurrent_hashmap"],
    difficulty=4,
    rubric_key="java.concurrency.concurrent_hashmap",
)

KAFKA_QUESTION = QuestionSpec(
    prompt_text="What ordering guarantees does Kafka give you?",
    interview_type=InterviewType.DISTRIBUTED,
    agent_key="distributed",
    concept_keys=["kafka.ordering"],
    difficulty=3,
    rubric_key="kafka.ordering",
)


def _routing(evaluation, concept_key: str, graph: SkillGraph | None = None, followups: int = 0):
    return RoutingContext(
        concept_key=concept_key,
        evaluation=evaluation,
        skill_graph=graph or SkillGraph(),
        followups_on_concept=followups,
        questions_asked=2,
        questions_remaining=6,
        minutes_remaining=12.0,
        available_concepts=[concept_key],
    )


class TestExcellentAnswer:
    def test_a_strong_answer_scores_high(self):
        evaluation = EvaluationEngine().evaluate(
            HASHMAP_QUESTION, AnswerPayload(text=STRONG_HASHMAP_ANSWER, self_confidence=5)
        )
        assert evaluation.score >= 0.8
        assert not evaluation.misconception.detected
        assert "treeification" in evaluation.demonstrated

    def test_a_strong_answer_makes_the_interview_go_deeper(self):
        """Spec section 2: do not keep asking basic questions after a strong answer."""
        evaluation = EvaluationEngine().evaluate(
            HASHMAP_QUESTION, AnswerPayload(text=STRONG_HASHMAP_ANSWER, self_confidence=5)
        )
        decision = AdaptiveRouterAgent().decide(
            _routing(evaluation, "java.collections.hashmap")
        )
        assert decision.direction in {AdaptiveDirection.DEEPER, AdaptiveDirection.HARDER}
        assert decision.difficulty_delta >= 1

    def test_descent_follows_the_taxonomy_edge(self):
        evaluation = EvaluationEngine().evaluate(
            HASHMAP_QUESTION, AnswerPayload(text=STRONG_HASHMAP_ANSWER)
        )
        decision = AdaptiveRouterAgent().decide(
            _routing(evaluation, "java.collections.hashmap")
        )
        if decision.direction is AdaptiveDirection.DEEPER:
            assert decision.next_concept_key == "java.concurrency.concurrent_hashmap"

    def test_a_saturated_concept_is_not_asked_again(self):
        graph = SkillGraph()
        for _ in range(3):
            graph.record(["java.collections.hashmap"], Evidence(score=0.95, difficulty=4))
        assert graph.is_saturated("java.collections.hashmap")

        evaluation = EvaluationEngine().evaluate(
            HASHMAP_QUESTION, AnswerPayload(text=STRONG_HASHMAP_ANSWER)
        )
        decision = AdaptiveRouterAgent().decide(
            _routing(evaluation, "java.collections.hashmap", graph)
        )
        assert decision.direction is not AdaptiveDirection.PROBE


class TestConfidentlyIncorrectAnswer:
    def test_misconception_is_detected(self):
        """Spec section 22: the low-knowledge/high-confidence quadrant."""
        answer = AnswerPayload(
            text="ConcurrentHashMap is basically a synchronized HashMap; it locks the "
            "whole map so only one thread can touch it.",
            self_confidence=5,
        )
        evaluation = EvaluationEngine().evaluate(CHM_QUESTION, answer)
        assert evaluation.misconception.detected
        assert "synchronized HashMap" in evaluation.misconception.belief
        assert evaluation.misconception.correction

    def test_a_misconception_caps_the_score(self):
        answer = AnswerPayload(
            text="ConcurrentHashMap is basically a synchronized HashMap; it locks the "
            "whole map so only one thread can touch it.",
            self_confidence=5,
        )
        evaluation = EvaluationEngine().evaluate(CHM_QUESTION, answer)
        assert evaluation.score <= 0.45

    def test_a_misconception_triggers_a_probe(self):
        """Spec section 5: probe rather than correct."""
        answer = AnswerPayload(text=CONFIDENTLY_WRONG_KAFKA, self_confidence=5)
        evaluation = EvaluationEngine().evaluate(KAFKA_QUESTION, answer)
        assert evaluation.misconception.detected

        decision = AdaptiveRouterAgent().decide(_routing(evaluation, "kafka.ordering"))
        assert decision.direction is AdaptiveDirection.PROBE
        assert decision.next_concept_key == "kafka.ordering"

    def test_the_probe_does_not_reveal_the_answer(self):
        answer = AnswerPayload(text=CONFIDENTLY_WRONG_KAFKA, self_confidence=5)
        evaluation = EvaluationEngine().evaluate(KAFKA_QUESTION, answer)

        probe = InterviewerAgent().probe(
            QuestionTarget(
                concept_keys=["kafka.ordering"],
                interview_type=InterviewType.DISTRIBUTED,
                difficulty=3,
                rubric_key="kafka.ordering",
            ),
            evaluation,
            question_text=KAFKA_QUESTION.prompt_text,
            answer_text=answer.text,
            asked_prompts=[KAFKA_QUESTION.prompt_text],
        )
        assert probe.is_followup
        lowered = probe.prompt_text.lower()
        # It must not simply tell them they are wrong.
        assert "you are wrong" not in lowered
        assert "incorrect" not in lowered
        assert "actually," not in lowered
        assert probe.prompt_text.strip().endswith("?") or len(probe.prompt_text) > 30

    def test_follow_up_budget_is_enforced(self):
        answer = AnswerPayload(text=CONFIDENTLY_WRONG_KAFKA, self_confidence=5)
        evaluation = EvaluationEngine().evaluate(KAFKA_QUESTION, answer)
        decision = AdaptiveRouterAgent().decide(
            _routing(evaluation, "kafka.ordering", followups=2)
        )
        # Two probes is enough; a third would just be badgering.
        assert decision.direction is not AdaptiveDirection.PROBE

    def test_hedged_wrongness_is_a_gap_not_a_misconception(self):
        """"I think maybe..." is an honest guess, and must not be reported as a
        confidently-held false belief."""
        answer = AnswerPayload(
            text="I think maybe Kafka guarantees ordering across the topic, but I'm not sure.",
            self_confidence=2,
        )
        evaluation = EvaluationEngine().evaluate(KAFKA_QUESTION, answer)
        assert not evaluation.misconception.detected


class TestWeakAndHonestAnswers:
    def test_a_weak_answer_makes_the_interview_back_off(self):
        evaluation = EvaluationEngine().evaluate(
            KAFKA_QUESTION, AnswerPayload(text="You just use Kafka and it works.")
        )
        decision = AdaptiveRouterAgent().decide(_routing(evaluation, "kafka.ordering"))
        assert decision.direction in {
            AdaptiveDirection.EASIER,
            AdaptiveDirection.PROBE,
            AdaptiveDirection.LATERAL,
        }
        assert decision.difficulty_delta <= 0

    def test_dont_know_is_classified_honestly(self):
        classification = ResponseClassifierAgent().classify(
            KAFKA_QUESTION, AnswerPayload(text="I don't know, I've never used Kafka.")
        )
        assert classification.response_class is ResponseClass.DONT_KNOW

    def test_a_clarifying_question_is_not_graded_as_an_answer(self):
        classification = ResponseClassifierAgent().classify(
            KAFKA_QUESTION,
            AnswerPayload(text="Quick question, should I assume a single region?"),
        )
        assert classification.response_class is ResponseClass.CLARIFYING_QUESTION

    def test_code_submissions_are_detected(self):
        classification = ResponseClassifierAgent().classify(
            KAFKA_QUESTION,
            AnswerPayload(text="def solve(nums):\n    return sorted(nums)"),
        )
        assert classification.response_class is ResponseClass.CODE_SUBMISSION
        assert classification.contains_code

    def test_short_answers_get_low_grader_confidence(self):
        """The grader should admit when there was little to grade."""
        thin = EvaluationEngine().evaluate(KAFKA_QUESTION, AnswerPayload(text="Per partition."))
        thorough = EvaluationEngine().evaluate(
            HASHMAP_QUESTION, AnswerPayload(text=STRONG_HASHMAP_ANSWER)
        )
        assert thin.confidence < thorough.confidence


class TestInterviewerConduct:
    def test_the_interviewer_never_repeats_a_question(self):
        interviewer = InterviewerAgent()
        target = QuestionTarget(
            concept_keys=["java.collections.hashmap"],
            interview_type=InterviewType.JAVA,
            difficulty=3,
            rubric_key="java.collections.hashmap",
        )
        asked: list[str] = []
        for _ in range(4):
            spec = interviewer.ask(target, asked_prompts=asked)
            assert spec.prompt_text not in asked
            asked.append(spec.prompt_text)

    def test_resume_probes_are_grounded_in_the_candidates_own_claim(self):
        claim = "Reduced p99 API latency by 35% by introducing a Redis read-through cache."
        spec = InterviewerAgent().ask(
            QuestionTarget(
                concept_keys=["system_design.caching"],
                interview_type=InterviewType.SYSTEM_DESIGN,
                difficulty=4,
                rubric_key="resume.claim_defense",
                is_resume_probe=True,
                claim_text=claim,
                claim_has_metric=True,
            ),
            asked_prompts=[],
            candidate_context=f"Resume claim under discussion: {claim}",
        )
        assert claim in spec.prompt_text
        assert spec.agent_key == "resume_defense"

    def test_the_persona_matches_the_concept(self):
        interviewer = InterviewerAgent()
        java = interviewer.ask(
            QuestionTarget(
                concept_keys=["java.collections.hashmap"],
                interview_type=InterviewType.JAVA,
                difficulty=3,
            ),
            asked_prompts=[],
        )
        kafka = interviewer.ask(
            QuestionTarget(
                concept_keys=["kafka.ordering"],
                interview_type=InterviewType.DISTRIBUTED,
                difficulty=3,
            ),
            asked_prompts=[],
        )
        assert java.agent_key == "java"
        assert kafka.agent_key == "distributed"
