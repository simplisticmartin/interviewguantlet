"""Shared Pydantic contracts.

Every agent boundary in Gauntlet is typed. An LLM never hands back free-form text
that another component has to guess at - it returns one of these models, validated
before it is allowed to influence the interview (spec section 41).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

Score = Annotated[float, Field(ge=0.0, le=1.0)]
Difficulty = Annotated[int, Field(ge=1, le=5)]
SelfConfidence = Annotated[int, Field(ge=1, le=5)]


class InterviewType(StrEnum):
    DSA = "dsa"
    JAVA = "java"
    SPRING = "spring"
    DATABASE = "database"
    DISTRIBUTED = "distributed"
    SYSTEM_DESIGN = "system_design"
    CLOUD = "cloud"
    FRONTEND = "frontend"
    AI_ENGINEERING = "ai_engineering"
    BEHAVIORAL = "behavioral"
    HIRING_MANAGER = "hiring_manager"
    RESUME_DEFENSE = "resume_defense"


class InterviewMode(StrEnum):
    REAL = "real"
    COACHING = "coaching"
    RAPID_FIRE = "rapid_fire"
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    RESUME_DEFENSE = "resume_defense"
    BEHAVIORAL = "behavioral"
    FULL_LOOP = "full_loop"


class AdaptiveDirection(StrEnum):
    """What the router decided to do with the next question (spec section 2)."""

    DEEPER = "deeper"  # same topic, one layer down
    HARDER = "harder"  # new topic, higher difficulty
    EASIER = "easier"  # back off, find the floor
    LATERAL = "lateral"  # adjacent topic, same difficulty
    PROBE = "probe"  # adversarial follow-up on a suspect claim
    MOVE_ON = "move_on"  # enough evidence here


class StrictModel(BaseModel):
    """Base for LLM-facing structures: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Candidate profile / resume
# ---------------------------------------------------------------------------


class ResumeProject(StrictModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class ResumeClaimModel(StrictModel):
    """A specific, checkable assertion from the resume (spec section 14)."""

    claim_text: str
    claim_type: str = "experience"
    technologies: list[str] = Field(default_factory=list)
    concept_keys: list[str] = Field(default_factory=list)
    has_metric: bool = False
    probe_priority: Annotated[int, Field(ge=1, le=5)] = 3


class ResumeProfile(StrictModel):
    display_name: str = "Candidate"
    headline: str = ""
    years_experience: Annotated[float, Field(ge=0, le=60)] = 0.0
    primary_languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    concept_keys: list[str] = Field(default_factory=list)
    projects: list[ResumeProject] = Field(default_factory=list)
    claims: list[ResumeClaimModel] = Field(default_factory=list)


class WeightedConcept(StrictModel):
    concept_key: str
    weight: Score = 0.5
    reason: str = ""


class JobAnalysis(StrictModel):
    title: str = "Software Engineer"
    level: str = "senior"
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    weighted_concepts: list[WeightedConcept] = Field(default_factory=list)
    domain: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class FocusArea(StrictModel):
    interview_type: InterviewType
    weight: Score
    concept_keys: list[str] = Field(default_factory=list)
    rationale: str = ""


class InterviewPlan(StrictModel):
    focus_areas: list[FocusArea] = Field(default_factory=list)
    opening_difficulty: Difficulty = 3
    target_question_count: Annotated[int, Field(ge=1, le=40)] = 8
    resume_claims_to_probe: list[str] = Field(default_factory=list)
    rationale: str = ""
    is_company_estimated: bool = True

    @field_validator("focus_areas")
    @classmethod
    def _at_least_one_area(cls, value: list[FocusArea]) -> list[FocusArea]:
        if not value:
            raise ValueError("interview plan needs at least one focus area")
        return value

    def normalised_weights(self) -> dict[InterviewType, float]:
        total = sum(area.weight for area in self.focus_areas)
        if total <= 0:
            share = 1.0 / len(self.focus_areas)
            return {area.interview_type: share for area in self.focus_areas}
        return {area.interview_type: area.weight / total for area in self.focus_areas}


# ---------------------------------------------------------------------------
# Questions and answers
# ---------------------------------------------------------------------------


class QuestionSpec(StrictModel):
    """A question the interviewer intends to ask."""

    prompt_text: str
    interview_type: InterviewType
    agent_key: str
    concept_keys: list[str] = Field(default_factory=list)
    difficulty: Difficulty = 3
    rubric_key: str | None = None
    is_followup: bool = False
    probe_reason: str | None = None
    asks_confidence: bool = False
    expects_code: bool = False
    source_question_id: str | None = None

    @field_validator("prompt_text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question prompt cannot be empty")
        return value


class AnswerPayload(StrictModel):
    text: str = ""
    code: str | None = None
    language: str | None = None
    self_confidence: SelfConfidence | None = None
    latency_ms: int | None = None


class ResponseClass(StrEnum):
    SUBSTANTIVE = "substantive"
    CLARIFYING_QUESTION = "clarifying_question"
    DONT_KNOW = "dont_know"
    CODE_SUBMISSION = "code_submission"
    OFF_TOPIC = "off_topic"
    EMPTY = "empty"


class ResponseClassification(StrictModel):
    response_class: ResponseClass
    contains_code: bool = False
    detected_language: str | None = None
    asks_for_clarification: bool = False
    clarification_text: str | None = None


class CoachingNote(StrictModel):
    """Teaching delivered between questions in Coaching Mode (spec section 25).

    This is the one place the system is allowed to tell a candidate they were wrong
    mid-interview. Real Interview Mode never produces one - measuring someone changes
    the measurement the moment you start teaching into it.
    """

    feedback: str
    key_correction: str | None = None
    next_step_hint: str | None = None


class ClarificationReply(StrictModel):
    """The interviewer's answer to a candidate's clarifying question.

    Answering these is legitimate even in Real Interview Mode - real interviewers do it
    constantly. ``gave_away_answer`` is the guard: a clarification that leaks the thing
    being assessed must be flagged so it can be discounted from the evidence weight.
    """

    reply: str
    restates_question: bool = False
    gave_away_answer: bool = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class RubricDimension(StrictModel):
    key: str
    label: str
    weight: float = 1.0
    hint: str = ""
    # Surface forms that indicate the candidate actually touched this dimension.
    # Used by the offline heuristic judge and as a retrieval aid; the LLM judges
    # read `label`/`hint` and are not restricted to these strings.
    markers: list[str] = Field(default_factory=list)


class MisconceptionPattern(StrictModel):
    """A known false belief for a concept, with the phrasing that betrays it.

    Encoding these explicitly means the highest-value finding in the product
    (confidently wrong) does not depend solely on an LLM noticing.
    """

    belief: str
    correction: str
    markers: list[str] = Field(default_factory=list)
    negative_markers: list[str] = Field(default_factory=list)
    severity: Annotated[int, Field(ge=1, le=5)] = 3
    concept_key: str | None = None


class RubricSpec(StrictModel):
    key: str
    version: int = 1
    title: str
    concept_key: str | None = None
    dimensions: list[RubricDimension]
    common_misconceptions: list[MisconceptionPattern] = Field(default_factory=list)

    def dimension_keys(self) -> list[str]:
        return [d.key for d in self.dimensions]

    def total_weight(self) -> float:
        return sum(d.weight for d in self.dimensions) or 1.0


class JudgeVerdict(StrictModel):
    """One judge's structured opinion (spec sections 18 and 19)."""

    judge_key: str = "technical_accuracy"
    score: Score
    demonstrated: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    incorrect: list[str] = Field(default_factory=list)
    communication_score: Score | None = None
    confidence: Score = 0.5
    evidence_quotes: list[str] = Field(default_factory=list)
    notes: str = ""


class MisconceptionFinding(StrictModel):
    detected: bool = False
    concept_key: str | None = None
    belief: str = ""
    correction: str = ""
    evidence_quote: str | None = None
    severity: Annotated[int, Field(ge=1, le=5)] = 3


class AggregateEvaluation(StrictModel):
    """Judges merged into the single verdict the skill graph consumes."""

    score: Score
    communication_score: Score | None = None
    confidence: Score = 0.5
    demonstrated: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    incorrect: list[str] = Field(default_factory=list)
    verdicts: list[JudgeVerdict] = Field(default_factory=list)
    disagreement: float = 0.0
    misconception: MisconceptionFinding = Field(default_factory=MisconceptionFinding)


class AdaptiveDecision(StrictModel):
    direction: AdaptiveDirection
    next_concept_key: str | None = None
    reason: str = ""
    probe_prompt: str | None = None
    difficulty_delta: Annotated[int, Field(ge=-2, le=2)] = 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class SkillReading(StrictModel):
    concept_key: str
    display_name: str
    mastery: Score
    confidence: Score
    evidence_count: int = 0
    self_confidence: float | None = None
    is_misconception: bool = False


class CommitteeVerdict(StrictModel):
    recommendation: str = "NO_DECISION"
    scores: dict[str, float] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    most_likely_rejection_reason: str = ""


class StudyPlanItemModel(StrictModel):
    priority: Annotated[int, Field(ge=1)]
    concept_key: str
    title: str
    rationale: str = ""
    learn_items: list[str] = Field(default_factory=list)
    practice_items: list[dict[str, Any]] = Field(default_factory=list)
    reattempt_prompt: str | None = None


class StudyPlanModel(StrictModel):
    summary: str = ""
    items: list[StudyPlanItemModel] = Field(default_factory=list)


class ReplayMoment(StrictModel):
    ordinal: int
    at_minute: float
    prompt_text: str
    concept_key: str | None = None
    score: Score
    note: str = ""
    checkpoint_id: str | None = None


class Scorecard(StrictModel):
    """The candidate-facing result of an interview (spec section 28)."""

    overall: Annotated[int, Field(ge=0, le=100)]
    category_scores: dict[str, int] = Field(default_factory=dict)
    strongest_areas: list[SkillReading] = Field(default_factory=list)
    weakest_areas: list[SkillReading] = Field(default_factory=list)
    misconceptions: list[MisconceptionFinding] = Field(default_factory=list)
    resume_claims_tested: list[dict[str, Any]] = Field(default_factory=list)
    communication_notes: list[str] = Field(default_factory=list)
    missed_opportunities: list[str] = Field(default_factory=list)
    committee: CommitteeVerdict = Field(default_factory=CommitteeVerdict)
    study_plan: StudyPlanModel = Field(default_factory=StudyPlanModel)
    replay_moments: list[ReplayMoment] = Field(default_factory=list)
    questions_asked: int = 0
    duration_minutes: float = 0.0
