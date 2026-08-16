"""HTTP request and response models.

Separate from the domain schemas in ``gauntlet.schemas`` on purpose: the wire format is
allowed to evolve without dragging the interview engine with it, and it lets the API
withhold interviewer-only fields (rubrics, concept keys, probe reasons) that would
change candidate behaviour if exposed.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, EmailStr, Field

from gauntlet.schemas import InterviewMode, InterviewType


class RegisterRequest(BaseModel):
    email: EmailStr
    password: Annotated[str, Field(min_length=10, max_length=200)]
    display_name: Annotated[str, Field(min_length=1, max_length=200)] = "Candidate"


class LoginRequest(BaseModel):
    email: EmailStr
    password: Annotated[str, Field(min_length=1, max_length=200)]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    candidate_id: uuid.UUID
    display_name: str


class ResumeSummary(BaseModel):
    id: uuid.UUID
    filename: str
    created_at: datetime
    is_primary: bool
    years_experience: float = 0.0
    concept_count: int = 0
    claim_count: int = 0


class ResumeDetail(ResumeSummary):
    profile: dict[str, Any]
    excerpt: str


class ResumeTextRequest(BaseModel):
    """Paste-in alternative to file upload."""

    filename: Annotated[str, Field(max_length=200)] = "pasted-resume.txt"
    text: Annotated[str, Field(min_length=50, max_length=60_000)]


class JobAnalyzeRequest(BaseModel):
    text: Annotated[str, Field(min_length=50, max_length=60_000)]
    company: Annotated[str | None, Field(max_length=120)] = None
    title: Annotated[str | None, Field(max_length=240)] = None
    level: Annotated[str | None, Field(max_length=60)] = None


class JobAnalyzeResponse(BaseModel):
    id: uuid.UUID
    title: str
    level: str
    must_have: list[str]
    weighted_concepts: list[dict[str, Any]]
    summary: str


class CreateInterviewRequest(BaseModel):
    resume_id: uuid.UUID | None = None
    job_description_id: uuid.UUID | None = None
    target_role: Annotated[str, Field(min_length=2, max_length=200)] = "Senior Software Engineer"
    target_level: Annotated[str, Field(max_length=60)] = "senior"
    company: Annotated[str | None, Field(max_length=120)] = None
    mode: InterviewMode = InterviewMode.REAL
    interview_types: list[InterviewType] = Field(default_factory=list)
    minutes: Annotated[int, Field(ge=5, le=120)] = 20


class AnswerRequest(BaseModel):
    text: Annotated[str, Field(max_length=20_000)] = ""
    code: Annotated[str | None, Field(max_length=40_000)] = None
    language: Annotated[str | None, Field(max_length=40)] = None
    self_confidence: Annotated[int | None, Field(ge=1, le=5)] = None
    latency_ms: Annotated[int | None, Field(ge=0)] = None


class QuestionView(BaseModel):
    """What the candidate is allowed to see. No rubric, no concept, no score."""

    ordinal: int | None = None
    prompt_text: str | None = None
    interview_type: str | None = None
    expects_code: bool = False
    asks_confidence: bool = False
    is_followup: bool = False
    asked_at: str | None = None


class TurnResponse(BaseModel):
    session_id: uuid.UUID
    status: str
    question: QuestionView | None = None
    clarification: dict[str, Any] | None = None
    coaching: dict[str, Any] | None = None
    scorecard: dict[str, Any] | None = None
    remaining_seconds: int = 0
    questions_asked: int = 0


class InterviewSummary(BaseModel):
    id: uuid.UUID
    target_role: str
    target_level: str
    mode: str
    status: str
    company: str | None = None
    planned_minutes: int
    questions_asked: int = 0
    overall: int | None = None
    recommendation: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class TranscriptEntry(BaseModel):
    ordinal: int
    prompt_text: str
    interview_type: str
    is_followup: bool
    answer_text: str | None = None
    self_confidence: int | None = None
    # Populated only once the interview is complete - never mid-interview.
    score: float | None = None
    concept_keys: list[str] = Field(default_factory=list)


class InterviewDetail(InterviewSummary):
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    scorecard: dict[str, Any] | None = None
    plan: dict[str, Any] = Field(default_factory=dict)


class SkillView(BaseModel):
    concept_key: str
    display_name: str
    mastery: float
    confidence: float
    evidence_count: int
    self_confidence: float | None = None
    calibration: str = "unknown"
    due_at: datetime | None = None


class QuestionSearchResult(BaseModel):
    id: uuid.UUID | None = None
    slug: str | None = None
    question: str
    interview_type: str
    concept_keys: list[str]
    topics: list[str]
    difficulty: int
    rubric_key: str | None = None
    question_origin: str
    source_type: str
    score: float | None = None


class CompanyView(BaseModel):
    slug: str
    name: str
    sector: str
    aliases: list[str] = Field(default_factory=list)


class CompanyPatterns(BaseModel):
    slug: str
    name: str
    sector: str
    # Always present, always honest about where the numbers came from.
    evidence: str
    basis: str
    disclaimer: str
    distribution: dict[str, float]
    readiness: dict[str, Any] | None = None


class StudyPlanItemView(BaseModel):
    priority: int
    concept_key: str
    display_name: str
    title: str
    rationale: str
    learn_items: list[str]
    practice_items: list[dict[str, Any]]
    status: str


class StudyPlanView(BaseModel):
    id: uuid.UUID | None = None
    summary: str = ""
    created_at: datetime | None = None
    items: list[StudyPlanItemView] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    interviews_completed: int
    questions_answered: int
    average_overall: float | None = None
    readiness: list[SkillView]
    strongest: list[SkillView]
    weakest: list[SkillView]
    open_misconceptions: list[dict[str, Any]]
    improvement: list[dict[str, Any]]
    due_for_review: list[SkillView]
    confidence_calibration: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    version: str
    database: bool
    llm_provider: str
    llm_degraded: bool
    durable_checkpoints: bool
    tracing: bool = False
    semantic_embeddings: bool


# --- User-contributed questions (spec sections 37, 38) --------------------------


class ContributionRequest(BaseModel):
    """A question someone was actually asked, offered to the corpus.

    Free-text fields are length-capped at the edge. The screening filter is the real
    defence, but a cap keeps a pathological payload from reaching it at all.
    """

    question: Annotated[str, Field(min_length=15, max_length=2000)]
    company: Annotated[str | None, Field(max_length=120)] = None
    role: Annotated[str | None, Field(max_length=120)] = None
    level: Annotated[str | None, Field(max_length=60)] = None
    interview_round: Annotated[str | None, Field(max_length=60)] = None
    asked_on: date | None = None
    notes: Annotated[str | None, Field(max_length=2000)] = None
    difficulty: Annotated[int | None, Field(ge=1, le=5)] = None


class ContributionResponse(BaseModel):
    """What the contributor is told.

    Deliberately states that nothing is published yet. A contributor who thinks their
    question went live and then cannot find it will assume the feature is broken.
    """

    id: uuid.UUID
    status: str
    question: str
    concept_keys: list[str]
    interview_type: str | None
    difficulty: int
    safety_verdict: str
    review_reasons: list[str]
    duplicate_of: str | None = None
    message: str


class ContributionRejected(BaseModel):
    detail: str
    reasons: list[str]


class SubmissionView(BaseModel):
    """A queued submission as a moderator sees it."""

    id: uuid.UUID
    question: str
    company_slug: str | None
    level: str | None
    interview_type: str | None
    concept_keys: list[str]
    difficulty: int
    status: str
    safety_verdict: str
    safety_findings: list[str]
    review_reasons: list[str]
    near_duplicates: list[Any]
    duplicate_of_slug: str | None
    created_at: datetime
    published_question_id: uuid.UUID | None = None


class ModerationDecision(BaseModel):
    """A moderator's ruling, with optional corrections to the automatic tagging."""

    decision: Literal["approve", "reject"]
    note: Annotated[str | None, Field(max_length=1000)] = None
    interview_type: Annotated[str | None, Field(max_length=60)] = None
    concept_keys: Annotated[list[str] | None, Field(max_length=10)] = None
    difficulty: Annotated[int | None, Field(ge=1, le=5)] = None
