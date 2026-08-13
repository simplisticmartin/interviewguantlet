"""Deterministic offline provider.

This is not a stub. It is a rule-based interviewer and grader that reads the same
structured context the LLM prompts receive and produces valid, coherent results with
no network access. It exists for three real reasons:

1. A fresh clone runs an end-to-end interview before anyone adds an API key.
2. Tests assert on interview *behaviour* (does difficulty rise after a strong answer?)
   without paying for or flaking on model calls.
3. It is the heuristic baseline the LLM judges are measured against in ``/evals`` -
   an evaluator that cannot beat keyword matching is not earning its cost.

Its limits are real and are surfaced in the UI: it matches surface forms, so it cannot
tell insight from vocabulary. Never present its scores as a model-quality evaluation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from gauntlet.llm.base import LLMProvider, LLMRole, StructuredOutputError, StructuredResult, Usage
from gauntlet.llm.embeddings import get_embedder
from gauntlet.schemas import (
    AdaptiveDecision,
    AdaptiveDirection,
    ClarificationReply,
    CoachingNote,
    CommitteeVerdict,
    FocusArea,
    InterviewPlan,
    InterviewType,
    JobAnalysis,
    JudgeVerdict,
    MisconceptionFinding,
    QuestionSpec,
    ResponseClass,
    ResponseClassification,
    ResumeClaimModel,
    ResumeProfile,
    ResumeProject,
    StudyPlanItemModel,
    StudyPlanModel,
    WeightedConcept,
)

_CONTEXT_RE = re.compile(r"<context>\s*(.*?)\s*</context>", re.DOTALL)
_UNTRUSTED_RE = re.compile(
    r'<untrusted_data source="([^"]+)">\s*(.*?)\s*</untrusted_data>', re.DOTALL
)
_WORD = re.compile(r"[a-z0-9+#.]+")
_METRIC = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|x|ms|s|k|m|b|qps|rps|tps|gb|tb)\b", re.I)

_STOPWORDS = frozenset(
    (
        "a an and are as at be by for from has have how in into is it its of on or that "
        "the to was were what when which who why with you your they their this these "
        "those can could should would will do does did not no yes if then than there "
        "here about over under"
    ).split(" ")
)

_DONT_KNOW = re.compile(
    r"\b(i (don'?t|do not) know|no idea|not sure|never (used|worked|heard)|"
    r"can'?t remember|unfamiliar with|haven'?t (used|done))\b",
    re.I,
)
_CLARIFYING = re.compile(
    r"(\?\s*$)|^\s*(can you|could you|what do you mean|do you mean|should i assume|"
    r"just to clarify|quick question)",
    re.I,
)
_CODE_HINT = re.compile(
    r"(^|\n)\s*(def |class |public |private |function |const |let |var |import |"
    r"#include|return |for \(|while \()",
)


def _tokens(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def _phrase_present(phrase: str, haystack: str) -> bool:
    """Whole-phrase containment, tolerant of punctuation and separators."""
    normalised = re.sub(r"[\s_\-]+", " ", phrase.lower()).strip()
    if not normalised:
        return False
    pattern = r"\b" + r"[\s_\-]*".join(re.escape(part) for part in normalised.split()) + r"\b"
    return re.search(pattern, haystack) is not None


class ScriptedProvider(LLMProvider):
    """Rule-based provider. Same interface, zero network."""

    name = "scripted"

    def model_for(self, role: LLMRole) -> str:
        return f"scripted-heuristic-v1:{role.value}"

    def complete_structured[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        role: LLMRole = LLMRole.INTERVIEW,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
    ) -> StructuredResult[T]:
        context = _parse_context(user)
        untrusted = _parse_untrusted(user)

        handler = _HANDLERS.get(response_model.__name__)
        if handler is None:
            raise StructuredOutputError(
                f"scripted provider has no rule for {response_model.__name__}; "
                "add one or configure a real provider"
            )

        value = handler(context, untrusted)
        return StructuredResult(
            value=response_model.model_validate(value.model_dump()),
            model=self.model_for(role),
            provider=self.name,
            usage=Usage(),
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        role: LLMRole = LLMRole.INTERVIEW,
        temperature: float = 0.6,
        max_tokens: int = 1024,
    ) -> str:
        context = _parse_context(user)
        return str(context.get("fallback_text", "Let's move on to the next question."))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return get_embedder().embed(texts)


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------


def _parse_context(user: str) -> dict[str, Any]:
    match = _CONTEXT_RE.search(user)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_untrusted(user: str) -> dict[str, str]:
    return {source: content for source, content in _UNTRUSTED_RE.findall(user)}


def _answer_text(untrusted: dict[str, str]) -> str:
    for key in ("candidate_answer", "answer", "candidate_response"):
        if key in untrusted:
            return untrusted[key]
    return ""


def _claim_from_context(untrusted: dict[str, str]) -> str:
    """Pull the resume claim out of the fenced candidate-context block."""
    block = untrusted.get("candidate_context", "")
    match = re.search(r"Resume claim under discussion:\s*(.+)", block)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_resume_profile(context: dict[str, Any], untrusted: dict[str, str]) -> ResumeProfile:
    text = untrusted.get("resume", "")
    lowered = text.lower()
    taxonomy: list[dict[str, Any]] = context.get("taxonomy", [])

    matched_keys: list[str] = []
    languages: list[str] = []
    frameworks: list[str] = []
    for entry in taxonomy:
        surfaces = [entry.get("display_name", ""), *entry.get("aliases", [])]
        if any(_phrase_present(surface, lowered) for surface in surfaces if surface):
            key = entry["key"]
            matched_keys.append(key)
            domain = entry.get("domain", "")
            root = key.split(".")[0]
            if domain == "language" and root not in languages:
                languages.append(entry.get("display_name", root))
            elif domain == "framework":
                frameworks.append(entry.get("display_name", root))

    claims: list[ResumeClaimModel] = []
    for raw_line in re.split(r"[\n\r]+|(?<=[.;])\s+", text):
        line = raw_line.strip(" \t-*•")
        if len(line) < 25 or len(line) > 400:
            continue
        line_lower = line.lower()
        technologies = sorted(
            {
                entry.get("display_name", entry["key"])
                for entry in taxonomy
                if any(
                    _phrase_present(surface, line_lower)
                    for surface in [entry.get("display_name", ""), *entry.get("aliases", [])]
                    if surface
                )
            }
        )
        has_metric = bool(_METRIC.search(line))
        if not technologies and not has_metric:
            continue
        concept_keys = [
            entry["key"]
            for entry in taxonomy
            if entry.get("display_name") in technologies or entry["key"] in technologies
        ]
        priority = 5 if (has_metric and technologies) else 4 if has_metric else 3
        claims.append(
            ResumeClaimModel(
                claim_text=line[:400],
                claim_type="impact" if has_metric else "experience",
                technologies=technologies[:8],
                concept_keys=concept_keys[:6],
                has_metric=has_metric,
                probe_priority=priority,
            )
        )

    claims.sort(key=lambda claim: claim.probe_priority, reverse=True)

    years = 0.0
    year_match = re.search(r"(\d{1,2})\+?\s*(?:years|yrs)", lowered)
    if year_match:
        years = min(float(year_match.group(1)), 60.0)

    name = ""
    for line in text.splitlines():
        stripped = line.strip()
        if 2 <= len(stripped.split()) <= 4 and stripped.replace(" ", "").replace(".", "").isalpha():
            name = stripped
            break

    projects = [
        ResumeProject(name=claim.claim_text[:80], description=claim.claim_text[:300],
                      technologies=claim.technologies)
        for claim in claims[:4]
    ]

    return ResumeProfile(
        display_name=name or "Candidate",
        headline=(text.strip().splitlines() or [""])[0][:160],
        years_experience=years,
        primary_languages=languages[:6],
        frameworks=frameworks[:8],
        domains=[],
        concept_keys=sorted(set(matched_keys))[:40],
        projects=projects,
        claims=claims[:12],
    )


def _handle_job_analysis(context: dict[str, Any], untrusted: dict[str, str]) -> JobAnalysis:
    text = untrusted.get("job_description", "")
    lowered = text.lower()
    taxonomy: list[dict[str, Any]] = context.get("taxonomy", [])

    weighted: list[WeightedConcept] = []
    must_have: list[str] = []
    for entry in taxonomy:
        surfaces = [entry.get("display_name", ""), *entry.get("aliases", [])]
        hits = sum(1 for surface in surfaces if surface and _phrase_present(surface, lowered))
        if not hits:
            continue
        emphasis = 0.5 + min(hits, 3) * 0.15
        if re.search(r"(required|must have|strong)[^.\n]{0,80}"
                     + re.escape(entry.get("display_name", "").lower()), lowered):
            emphasis += 0.2
        weighted.append(
            WeightedConcept(
                concept_key=entry["key"],
                weight=round(min(emphasis, 1.0), 3),
                reason="mentioned in job description",
            )
        )
        must_have.append(entry.get("display_name", entry["key"]))

    weighted.sort(key=lambda item: item.weight, reverse=True)

    title_match = re.search(
        r"^(.*(?:engineer|developer|architect|scientist|manager).*)$", text, re.I | re.M
    )
    level = "senior"
    for candidate_level in ("principal", "staff", "senior", "junior", "mid", "lead"):
        if re.search(rf"\b{candidate_level}\b", lowered):
            level = candidate_level
            break

    return JobAnalysis(
        title=(title_match.group(1).strip()[:120] if title_match else "Software Engineer"),
        level=level,
        must_have=must_have[:12],
        nice_to_have=[],
        weighted_concepts=weighted[:24],
        domain="",
        summary=f"Matched {len(weighted)} assessable concepts from the posting.",
    )


def _handle_interview_plan(context: dict[str, Any], _untrusted: dict[str, str]) -> InterviewPlan:
    hints: list[dict[str, Any]] = context.get("focus_hints", [])
    areas: list[FocusArea] = []
    for hint in hints:
        try:
            areas.append(
                FocusArea(
                    interview_type=InterviewType(hint["interview_type"]),
                    weight=float(hint.get("weight", 0.2)),
                    concept_keys=list(hint.get("concept_keys", []))[:8],
                    rationale=str(hint.get("rationale", "")),
                )
            )
        except (KeyError, ValueError):
            continue

    if not areas:
        areas = [
            FocusArea(interview_type=InterviewType.JAVA, weight=0.4, rationale="default coverage"),
            FocusArea(interview_type=InterviewType.SYSTEM_DESIGN, weight=0.3,
                      rationale="default coverage"),
            FocusArea(interview_type=InterviewType.BEHAVIORAL, weight=0.3,
                      rationale="default coverage"),
        ]

    return InterviewPlan(
        focus_areas=areas,
        opening_difficulty=int(context.get("opening_difficulty", 3)),
        target_question_count=int(context.get("target_question_count", 8)),
        resume_claims_to_probe=list(context.get("resume_claims_to_probe", []))[:5],
        rationale="Heuristic plan: job-description concept weights intersected with resume skills.",
        is_company_estimated=bool(context.get("is_company_estimated", True)),
    )


def _handle_question_spec(context: dict[str, Any], untrusted: dict[str, str]) -> QuestionSpec:
    target: dict[str, Any] = context.get("target", {})
    asked: set[str] = {str(item) for item in context.get("asked_prompts", [])}

    # Resume cross-examination: the question must be grounded in the candidate's own
    # claim, walking the claim-defence rubric (measurement -> cause -> tradeoff -> scale)
    # rather than pulling an unrelated question from the corpus.
    claim = context.get("resume_claim") or _claim_from_context(untrusted)
    if context.get("is_resume_probe") and claim:
        dimensions = list(context.get("rubric_dimensions", []))
        if not context.get("claim_has_metric", True):
            # "How did you measure that?" is a non-question for a claim with no number
            # in it. Start from mechanism instead.
            dimensions = [d for d in dimensions if d.get("key") != "measurement"]
        for dimension in dimensions:
            probe = str(dimension.get("probe", "")).strip()
            if not probe:
                continue
            prompt_text = f'You wrote: "{str(claim).strip()}" {probe}'
            if prompt_text in asked:
                continue
            return QuestionSpec(
                prompt_text=prompt_text,
                interview_type=InterviewType(target.get("interview_type", "java")),
                agent_key="resume_defense",
                concept_keys=list(target.get("concept_keys", []))[:6],
                difficulty=int(target.get("difficulty", 3)),
                rubric_key=target.get("rubric_key"),
            )

    # Follow-up probe: aim at the first unmet rubric dimension, using its authored probe.
    for gap in context.get("gap_probes", []):
        if gap and gap not in asked:
            return QuestionSpec(
                prompt_text=str(gap),
                interview_type=InterviewType(target.get("interview_type", "java")),
                agent_key=str(target.get("agent_key", "java")),
                concept_keys=list(target.get("concept_keys", []))[:6],
                difficulty=int(target.get("difficulty", 3)),
                rubric_key=target.get("rubric_key"),
                is_followup=True,
                probe_reason=str(context.get("probe_reason", "gap in rubric coverage")),
            )

    for candidate in context.get("candidate_questions", []):
        prompt_text = str(candidate.get("prompt_text", "")).strip()
        if not prompt_text or prompt_text in asked:
            continue
        return QuestionSpec(
            prompt_text=prompt_text,
            interview_type=InterviewType(
                candidate.get("interview_type", target.get("interview_type", "java"))
            ),
            agent_key=str(candidate.get("agent_key", target.get("agent_key", "java"))),
            concept_keys=list(candidate.get("concept_keys", []))[:6],
            difficulty=int(candidate.get("difficulty", target.get("difficulty", 3))),
            rubric_key=candidate.get("rubric_key"),
            is_followup=False,
            asks_confidence=bool(candidate.get("asks_confidence", False)),
            expects_code=bool(candidate.get("expects_code", False)),
            source_question_id=candidate.get("id"),
        )

    concept_label = str(target.get("display_name") or target.get("concept_key") or "this area")
    return QuestionSpec(
        prompt_text=(
            f"Walk me through how {concept_label} works in a system you have built, "
            "and where it breaks down under load."
        ),
        interview_type=InterviewType(target.get("interview_type", "java")),
        agent_key=str(target.get("agent_key", "java")),
        concept_keys=list(target.get("concept_keys", []))[:6],
        difficulty=int(target.get("difficulty", 3)),
        rubric_key=target.get("rubric_key"),
    )


def _handle_response_classification(
    _context: dict[str, Any], untrusted: dict[str, str]
) -> ResponseClassification:
    text = _answer_text(untrusted)
    stripped = text.strip()

    if not stripped:
        return ResponseClassification(response_class=ResponseClass.EMPTY)

    contains_code = bool(_CODE_HINT.search(text)) or "```" in text
    if _DONT_KNOW.search(stripped) and len(stripped) < 240:
        return ResponseClassification(response_class=ResponseClass.DONT_KNOW)
    if contains_code:
        language = None
        if re.search(r"\b(public|private|static void|System\.out)\b", text):
            language = "java"
        elif re.search(r"\bdef |self\.|print\(", text):
            language = "python"
        elif re.search(r"\b(const|let|=>|console\.log)\b", text):
            language = "javascript"
        return ResponseClassification(
            response_class=ResponseClass.CODE_SUBMISSION,
            contains_code=True,
            detected_language=language,
        )
    if _CLARIFYING.search(stripped) and len(stripped) < 320:
        return ResponseClassification(
            response_class=ResponseClass.CLARIFYING_QUESTION,
            asks_for_clarification=True,
            clarification_text=stripped[:300],
        )
    return ResponseClassification(response_class=ResponseClass.SUBSTANTIVE)


def _handle_coaching(context: dict[str, Any], _untrusted: dict[str, str]) -> CoachingNote:
    evaluation: dict[str, Any] = context.get("last_evaluation", {})
    rubric: dict[str, Any] = context.get("rubric", {})
    labels = {
        str(dimension.get("key")): str(dimension.get("label", dimension.get("key")))
        for dimension in rubric.get("dimensions", [])
    }

    demonstrated = [labels.get(key, key) for key in evaluation.get("demonstrated", [])]
    missing = [labels.get(key, key) for key in evaluation.get("missing", [])]
    misconception: dict[str, Any] = evaluation.get("misconception") or {}

    parts: list[str] = []
    if demonstrated:
        parts.append(f"Good: you covered {', '.join(demonstrated[:3])}.")
    else:
        parts.append("There wasn't much there for me to work with yet.")

    correction: str | None = None
    if misconception.get("detected"):
        correction = str(misconception.get("correction", "")) or None
        parts.append(
            f"One thing to fix: you said \"{misconception.get('belief', '')}\" "
            f"That isn't right - {correction}"
        )
    elif missing:
        parts.append(f"The biggest gap was {missing[0]}.")

    hint: str | None = None
    for dimension in rubric.get("dimensions", []):
        if dimension.get("key") in set(evaluation.get("missing", [])) and dimension.get("hint"):
            hint = str(dimension["hint"])
            break

    return CoachingNote(
        feedback=" ".join(parts),
        key_correction=correction,
        next_step_hint=hint,
    )


def _handle_clarification(
    context: dict[str, Any], untrusted: dict[str, str]
) -> ClarificationReply:
    question = str(context.get("question", "")).lower()
    asked = _answer_text(untrusted).lower()

    # Scale questions get concrete numbers; anything else gets the standard interviewer
    # move of handing the assumption back to the candidate.
    if any(word in asked for word in ("scale", "how many", "volume", "qps", "traffic", "users")):
        reply = (
            "Assume roughly 10 million users and a few thousand requests per second at "
            "peak. Design for that, and tell me where it would break beyond it."
        )
    elif any(word in asked for word in ("input", "size", "constraint", "range", "null", "empty")):
        reply = (
            "Assume the input can be large and may be empty or contain duplicates. "
            "State whatever assumption you need and carry on."
        )
    elif "language" in asked or "which" in asked:
        reply = "Use whichever language you are most comfortable in."
    elif "design" in question or "architecture" in question:
        reply = (
            "That ambiguity is deliberate - part of what I am looking for is how you "
            "resolve it. State your assumption and proceed."
        )
    else:
        reply = (
            "Take whatever assumption seems reasonable to you, just say it out loud "
            "before you continue."
        )

    return ClarificationReply(reply=reply, restates_question=False, gave_away_answer=False)


def _handle_judge_verdict(context: dict[str, Any], untrusted: dict[str, str]) -> JudgeVerdict:
    answer = _answer_text(untrusted)
    lowered = re.sub(r"[\s_\-]+", " ", answer.lower())
    rubric: dict[str, Any] = context.get("rubric", {})
    dimensions: list[dict[str, Any]] = rubric.get("dimensions", [])
    judge_key = str(context.get("judge_key", "technical_accuracy"))

    demonstrated: list[str] = []
    missing: list[str] = []
    total_weight = 0.0
    earned = 0.0

    for dimension in dimensions:
        weight = float(dimension.get("weight", 1.0))
        total_weight += weight
        surfaces = list(dimension.get("markers", []))
        surfaces.append(str(dimension.get("key", "")).replace("_", " "))
        surfaces.append(str(dimension.get("label", "")))
        if any(surface and _phrase_present(surface, lowered) for surface in surfaces):
            demonstrated.append(str(dimension.get("key")))
            earned += weight
        else:
            missing.append(str(dimension.get("key")))

    incorrect: list[str] = []
    evidence: list[str] = []
    for pattern in rubric.get("common_misconceptions", []):
        markers = pattern.get("markers", [])
        negatives = pattern.get("negative_markers", [])
        if any(_phrase_present(marker, lowered) for marker in markers) and not any(
            _phrase_present(neg, lowered) for neg in negatives
        ):
            incorrect.append(str(pattern.get("concept_key") or pattern.get("belief", ""))[:80])
            evidence.append(str(pattern.get("belief", ""))[:200])

    score = (earned / total_weight) if total_weight else 0.0
    # A stated misconception is worse than silence: penalise, do not merely omit.
    score = max(0.0, score - 0.15 * len(incorrect))

    if _DONT_KNOW.search(answer) and len(answer.strip()) < 240:
        score = min(score, 0.1)

    word_count = len(answer.split())
    # Confidence in our own grading scales with how much there was to grade.
    judge_confidence = 0.25 if word_count < 15 else 0.5 if word_count < 60 else 0.7
    communication = min(1.0, 0.3 + min(word_count, 200) / 300) if word_count else 0.0
    if judge_key == "communication":
        score = communication

    return JudgeVerdict(
        judge_key=judge_key,
        score=round(min(max(score, 0.0), 1.0), 3),
        demonstrated=demonstrated,
        missing=missing,
        incorrect=incorrect,
        communication_score=round(communication, 3),
        confidence=judge_confidence,
        evidence_quotes=evidence,
        notes="Heuristic marker matching (offline provider); not a semantic judgement.",
    )


def _handle_misconception(
    context: dict[str, Any], untrusted: dict[str, str]
) -> MisconceptionFinding:
    answer = _answer_text(untrusted)
    lowered = re.sub(r"[\s_\-]+", " ", answer.lower())
    rubric: dict[str, Any] = context.get("rubric", {})

    hedged = bool(re.search(r"\b(i think|maybe|not sure|i believe|possibly|might be)\b", lowered))

    for pattern in rubric.get("common_misconceptions", []):
        markers = pattern.get("markers", [])
        negatives = pattern.get("negative_markers", [])
        if not any(_phrase_present(marker, lowered) for marker in markers):
            continue
        if any(_phrase_present(neg, lowered) for neg in negatives):
            continue
        if hedged:
            # Hedged wrongness is a knowledge gap, not a misconception (spec section 22).
            continue
        return MisconceptionFinding(
            detected=True,
            concept_key=pattern.get("concept_key") or context.get("concept_key"),
            belief=str(pattern.get("belief", "")),
            correction=str(pattern.get("correction", "")),
            evidence_quote=_first_matching_sentence(answer, markers),
            severity=int(pattern.get("severity", 3)),
        )

    return MisconceptionFinding(detected=False)


def _first_matching_sentence(text: str, markers: list[str]) -> str | None:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lowered = re.sub(r"[\s_\-]+", " ", sentence.lower())
        if any(_phrase_present(marker, lowered) for marker in markers):
            return sentence.strip()[:300]
    return None


def _handle_adaptive_decision(
    context: dict[str, Any], _untrusted: dict[str, str]
) -> AdaptiveDecision:
    evaluation: dict[str, Any] = context.get("last_evaluation", {})
    score = float(evaluation.get("score", 0.5))
    misconception = bool(evaluation.get("misconception_detected", False))
    followups_used = int(context.get("followups_on_concept", 0))
    concept_key = context.get("concept_key")
    deeper_options: list[str] = context.get("deeper_concepts", [])

    if misconception and followups_used < 2:
        return AdaptiveDecision(
            direction=AdaptiveDirection.PROBE,
            next_concept_key=concept_key,
            reason="Misconception detected - probing before it is scored as understood.",
            difficulty_delta=0,
        )
    if score >= 0.8:
        if deeper_options and followups_used < 2:
            return AdaptiveDecision(
                direction=AdaptiveDirection.DEEPER,
                next_concept_key=deeper_options[0],
                reason=f"Strong answer ({score:.2f}); descending to a harder sub-concept.",
                difficulty_delta=1,
            )
        return AdaptiveDecision(
            direction=AdaptiveDirection.HARDER,
            reason=f"Strong answer ({score:.2f}) and this concept is exhausted.",
            difficulty_delta=1,
        )
    if score < 0.35:
        return AdaptiveDecision(
            direction=AdaptiveDirection.EASIER,
            reason=f"Weak answer ({score:.2f}); locating the floor rather than piling on.",
            difficulty_delta=-1,
        )
    if score < 0.6 and followups_used < 1:
        return AdaptiveDecision(
            direction=AdaptiveDirection.PROBE,
            next_concept_key=concept_key,
            reason=f"Partial answer ({score:.2f}); one probe to separate gap from wording.",
        )
    return AdaptiveDecision(
        direction=AdaptiveDirection.LATERAL,
        reason=f"Adequate answer ({score:.2f}); moving to an adjacent area.",
    )


def _handle_committee(context: dict[str, Any], _untrusted: dict[str, str]) -> CommitteeVerdict:
    scores: dict[str, float] = {
        str(key): float(value) for key, value in context.get("category_scores", {}).items()
    }
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    strengths_src: list[dict[str, Any]] = context.get("strengths", [])
    weaknesses_src: list[dict[str, Any]] = context.get("weaknesses", [])
    misconceptions: list[dict[str, Any]] = context.get("misconceptions", [])

    if overall >= 0.85:
        recommendation = "STRONG_HIRE"
    elif overall >= 0.72:
        recommendation = "HIRE"
    elif overall >= 0.6:
        recommendation = "LEAN_HIRE"
    elif overall >= 0.45:
        recommendation = "LEAN_NO_HIRE"
    else:
        recommendation = "NO_HIRE"

    if misconceptions and recommendation in {"STRONG_HIRE", "HIRE"}:
        recommendation = "LEAN_HIRE"

    rejection = ""
    if misconceptions:
        rejection = (
            f"Confidently incorrect on {misconceptions[0].get('concept_key', 'a core concept')}: "
            f"{misconceptions[0].get('belief', '')}"
        )
    elif weaknesses_src:
        weakest = weaknesses_src[0]
        rejection = (
            f"Insufficient depth in {weakest.get('display_name', weakest.get('concept_key'))} "
            f"(mastery {float(weakest.get('mastery', 0)):.2f}) for the target level."
        )

    return CommitteeVerdict(
        recommendation=recommendation,
        scores={key: round(value, 3) for key, value in scores.items()},
        strengths=[
            f"{item.get('display_name', item.get('concept_key'))}: "
            f"mastery {float(item.get('mastery', 0)):.2f} over "
            f"{item.get('evidence_count', 0)} question(s)"
            for item in strengths_src[:5]
        ],
        risks=[
            f"{item.get('display_name', item.get('concept_key'))}: "
            f"mastery {float(item.get('mastery', 0)):.2f}"
            for item in weaknesses_src[:5]
        ],
        evidence=[str(item.get("belief", ""))[:200] for item in misconceptions[:5]],
        next_steps=[
            f"Close the gap in {item.get('display_name', item.get('concept_key'))}"
            for item in weaknesses_src[:3]
        ],
        most_likely_rejection_reason=rejection,
    )


def _handle_study_plan(context: dict[str, Any], _untrusted: dict[str, str]) -> StudyPlanModel:
    items: list[StudyPlanItemModel] = []
    priority = 1

    for misconception in context.get("misconceptions", [])[:3]:
        concept_key = str(misconception.get("concept_key") or "general")
        items.append(
            StudyPlanItemModel(
                priority=priority,
                concept_key=concept_key,
                title=f"Correct your model of {concept_key.split('.')[-1].replace('_', ' ')}",
                rationale=(
                    f"You stated this as fact during the interview: "
                    f"\"{misconception.get('belief', '')}\". "
                    f"The accurate statement is: {misconception.get('correction', '')}"
                ),
                learn_items=[str(misconception.get("correction", ""))],
                practice_items=[
                    {
                        "type": "question",
                        "prompt": f"Explain {concept_key.split('.')[-1].replace('_', ' ')} "
                        "and name one scenario where getting it wrong causes a production bug.",
                    }
                ],
                reattempt_prompt=misconception.get("reattempt_prompt"),
            )
        )
        priority += 1

    for weakness in context.get("weaknesses", [])[:4]:
        concept_key = str(weakness.get("concept_key", "general"))
        label = str(weakness.get("display_name", concept_key))
        items.append(
            StudyPlanItemModel(
                priority=priority,
                concept_key=concept_key,
                title=f"Build depth in {label}",
                rationale=(
                    f"Measured mastery {float(weakness.get('mastery', 0)):.2f} across "
                    f"{weakness.get('evidence_count', 0)} question(s), and the target role "
                    "weights this area highly."
                ),
                learn_items=list(weakness.get("sub_concepts", []))[:6],
                practice_items=[
                    {"type": "question", "prompt": prompt}
                    for prompt in list(weakness.get("practice_prompts", []))[:3]
                ],
                reattempt_prompt=weakness.get("reattempt_prompt"),
            )
        )
        priority += 1

    return StudyPlanModel(
        summary=(
            f"{len(items)} prioritised item(s), ordered by interview impact: "
            "confidently-wrong beliefs first, then role-critical gaps."
        ),
        items=items,
    )


_HANDLERS: dict[str, Callable[[dict[str, Any], dict[str, str]], BaseModel]] = {
    "ResumeProfile": _handle_resume_profile,
    "JobAnalysis": _handle_job_analysis,
    "InterviewPlan": _handle_interview_plan,
    "QuestionSpec": _handle_question_spec,
    "ResponseClassification": _handle_response_classification,
    "ClarificationReply": _handle_clarification,
    "CoachingNote": _handle_coaching,
    "JudgeVerdict": _handle_judge_verdict,
    "MisconceptionFinding": _handle_misconception,
    "AdaptiveDecision": _handle_adaptive_decision,
    "CommitteeVerdict": _handle_committee,
    "StudyPlanModel": _handle_study_plan,
}
