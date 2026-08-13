"""Final report: scorecard, hiring committee, study plan, replay moments."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from gauntlet.agents.committee import HiringCommitteeAgent
from gauntlet.agents.study import StudyPlannerAgent
from gauntlet.graph.state import InterviewState, elapsed_seconds, skill_graph_from_state
from gauntlet.schemas import (
    JobAnalysis,
    MisconceptionFinding,
    ReplayMoment,
    ResumeProfile,
    Scorecard,
)
from gauntlet.skills.graph import category_scores

log = structlog.get_logger(__name__)

# A moment worth replaying: a weak answer, or one where a misconception surfaced.
REPLAY_SCORE_THRESHOLD = 0.55


def build_report(state: InterviewState) -> dict[str, Any]:
    graph = skill_graph_from_state(state)
    readings = graph.leaf_readings()
    categories = category_scores(readings)

    misconceptions, repeat_counts = _dedupe_misconceptions(state.get("misconceptions", []))

    job = JobAnalysis.model_validate(state.get("job_description") or {})
    job_concept_keys = [item.concept_key for item in job.weighted_concepts]

    evidence_quotes = _evidence_quotes(state)
    asked = [item.get("prompt_text", "") for item in state.get("question_history", [])]

    committee = HiringCommitteeAgent().deliberate(
        readings=readings,
        category_scores=categories,
        misconceptions=misconceptions,
        evidence_quotes=evidence_quotes,
        target_role=state.get("target_role", "Software Engineer"),
        target_level=state.get("target_level", "senior"),
        company_name=(state.get("company_patterns") or {}).get("name"),
        questions_asked=len(asked),
    )

    study_plan = StudyPlannerAgent().build(
        readings=readings,
        misconceptions=misconceptions,
        job_concept_keys=job_concept_keys,
        asked_prompts=asked,
        target_role=state.get("target_role", "Software Engineer"),
        target_level=state.get("target_level", "senior"),
    )

    overall = _overall_score(categories, readings)
    duration = elapsed_seconds(state) / 60.0

    scorecard = Scorecard(
        overall=overall,
        category_scores=categories,
        strongest_areas=graph.strongest(limit=5),
        weakest_areas=graph.weakest(limit=5),
        misconceptions=[finding for finding in misconceptions if finding.detected],
        resume_claims_tested=_resume_claims_tested(state, graph),
        communication_notes=_communication_notes(state, misconceptions, repeat_counts),
        missed_opportunities=_missed_opportunities(state),
        committee=committee,
        study_plan=study_plan,
        replay_moments=_replay_moments(state),
        questions_asked=len(asked),
        duration_minutes=round(duration, 2),
    )

    log.info(
        "graph.report",
        session=state.get("session_id"),
        overall=overall,
        recommendation=committee.recommendation,
        questions=len(asked),
    )

    return {
        "final_scorecard": scorecard.model_dump(mode="json"),
        "status": "completed",
        "elapsed_time": elapsed_seconds(state),
        "remaining_time": 0,
        "current_question": None,
        "pending_target": None,
        "pending_answer": None,
    }


def _dedupe_misconceptions(
    rows: list[dict[str, Any]],
) -> tuple[list[MisconceptionFinding], dict[tuple[str, str], int]]:
    """Collapse repeats of the same belief, keeping how often it recurred.

    A candidate who restates the same wrong belief under three follow-ups has one
    misconception, not three - but the repetition is itself signal (spec section 35
    tracks repeated misconceptions), so the count is preserved and the severity of a
    belief they would not let go of is raised.
    """
    grouped: dict[tuple[str, str], MisconceptionFinding] = {}
    counts: dict[tuple[str, str], int] = {}

    for row in rows:
        payload = {
            key: value for key, value in row.items() if key in MisconceptionFinding.model_fields
        }
        finding = MisconceptionFinding.model_validate(payload)
        identity = (finding.concept_key or "", finding.belief.strip().lower())
        counts[identity] = counts.get(identity, 0) + 1
        if identity not in grouped:
            grouped[identity] = finding
        elif finding.evidence_quote and not grouped[identity].evidence_quote:
            grouped[identity] = grouped[identity].model_copy(
                update={"evidence_quote": finding.evidence_quote}
            )

    findings: list[MisconceptionFinding] = []
    for identity, finding in grouped.items():
        repeats = counts[identity]
        if repeats > 1:
            # Held under direct challenge: that is worse than saying it once.
            finding = finding.model_copy(update={"severity": min(5, finding.severity + 1)})
        findings.append(finding)

    def _rank(item: MisconceptionFinding) -> tuple[int, int]:
        identity = (item.concept_key or "", item.belief.strip().lower())
        return (-item.severity, -counts[identity])

    findings.sort(key=_rank)
    return findings, counts


def _overall_score(categories: dict[str, int], readings: list[Any]) -> int:
    if categories:
        return round(sum(categories.values()) / len(categories))
    scored = [reading for reading in readings if reading.evidence_count]
    if not scored:
        return 0
    return round(100 * sum(reading.mastery for reading in scored) / len(scored))


def _evidence_quotes(state: InterviewState) -> list[str]:
    """Verbatim candidate lines the judges flagged, for citation in the verdict."""
    quotes: list[str] = []
    for row in state.get("misconceptions", []):
        quote = row.get("evidence_quote") or row.get("belief")
        if quote:
            quotes.append(str(quote))
    evaluation = state.get("last_evaluation") or {}
    for verdict in evaluation.get("verdicts", []):
        quotes.extend(str(item) for item in verdict.get("evidence_quotes", []))
    return list(dict.fromkeys(quotes))


def _resume_claims_tested(state: InterviewState, graph: Any) -> list[dict[str, Any]]:
    """Which resume claims were probed, and how much evidence backs them.

    ``support`` describes evidence depth only. It is never a judgement about honesty
    (spec section 14).
    """
    profile = ResumeProfile.model_validate(state.get("resume_profile") or {})
    if not profile.claims:
        return []

    probed = {
        item.get("claim_text"): item
        for item in state.get("question_history", [])
        if item.get("is_resume_probe") and item.get("claim_text")
    }

    scores_by_ordinal = {
        item.get("ordinal"): item.get("score", 0.0)
        for item in state.get("answer_history", [])
    }

    rows: list[dict[str, Any]] = []
    for claim in profile.claims:
        question = probed.get(claim.claim_text)
        if question is None:
            rows.append(
                {
                    "claim": claim.claim_text,
                    "tested": False,
                    "support": "not_tested",
                    "score": None,
                }
            )
            continue
        score = float(scores_by_ordinal.get(question.get("ordinal"), 0.0))
        if score >= 0.75:
            support = "well_supported"
        elif score >= 0.5:
            support = "partially_supported"
        elif score >= 0.25:
            support = "thinly_supported"
        else:
            support = "little_evidence"
        rows.append(
            {
                "claim": claim.claim_text,
                "tested": True,
                "support": support,
                "score": round(score, 3),
                "question": question.get("prompt_text"),
            }
        )
    return rows


def _communication_notes(
    state: InterviewState,
    misconceptions: list[MisconceptionFinding] | None = None,
    repeat_counts: dict[tuple[str, str], int] | None = None,
) -> list[str]:
    notes: list[str] = []
    answers = state.get("answer_history", [])
    if not answers:
        return notes

    # Restating a belief after a direct challenge is a coachability signal, and it is
    # one candidates almost never get told about.
    for finding in misconceptions or []:
        identity = (finding.concept_key or "", finding.belief.strip().lower())
        repeats = (repeat_counts or {}).get(identity, 1)
        if repeats > 1:
            notes.append(
                f'You restated "{finding.belief}" {repeats} times, including after the '
                "interviewer probed it. When an interviewer circles back to the same "
                "point, treat it as an invitation to re-examine the assumption."
            )

    lengths = [len(str(item.get("text", "")).split()) for item in answers]
    average = sum(lengths) / len(lengths)
    if average < 25:
        notes.append(
            f"Answers averaged {average:.0f} words. Short answers give an interviewer "
            "little to score - state your reasoning, not just your conclusion."
        )
    elif average > 220:
        notes.append(
            f"Answers averaged {average:.0f} words. Lead with the answer, then expand; "
            "interviewers are listening for structure."
        )

    dont_knows = sum(1 for item in answers if item.get("response_class") == "dont_know")
    if dont_knows and dont_knows / len(answers) > 0.3:
        notes.append(
            f"{dont_knows} of {len(answers)} answers were explicit concessions. Honest, "
            "but try reasoning aloud toward a partial answer before conceding."
        )
    return notes


def _missed_opportunities(state: InterviewState) -> list[str]:
    """Rubric dimensions the candidate never touched - the cheapest score to recover."""
    evaluation = state.get("last_evaluation") or {}
    missed: list[str] = []

    misses = evaluation.get("missing", [])
    if misses:
        missed.append(
            "On the final question you did not address: " + ", ".join(str(m) for m in misses[:5])
        )

    code_check = state.get("code_check") or {}
    missed.extend(str(signal) for signal in code_check.get("interviewer_signals", []))
    return missed


def _replay_moments(state: InterviewState) -> list[ReplayMoment]:
    """The points worth restoring the interview to and attempting again."""
    started = state.get("started_at")
    try:
        start = datetime.fromisoformat(started) if started else None
    except ValueError:
        start = None

    scores = {item.get("ordinal"): item for item in state.get("answer_history", [])}
    misconception_ordinals = {
        row.get("ordinal"): row for row in state.get("misconceptions", [])
    }

    moments: list[ReplayMoment] = []
    for question in state.get("question_history", []):
        ordinal = question.get("ordinal")
        answer = scores.get(ordinal)
        if answer is None:
            continue
        score = float(answer.get("score", 0.0))
        misconception = misconception_ordinals.get(ordinal)
        if score > REPLAY_SCORE_THRESHOLD and not misconception:
            continue

        at_minute = 0.0
        asked_at = question.get("asked_at")
        if start and asked_at:
            try:
                moment_time = datetime.fromisoformat(str(asked_at))
                if moment_time.tzinfo is None:
                    moment_time = moment_time.replace(tzinfo=UTC)
                at_minute = max(0.0, (moment_time - start).total_seconds() / 60.0)
            except ValueError:
                at_minute = 0.0

        note = (
            f"Misconception surfaced: {misconception.get('belief', '')}"
            if misconception
            else "Weak answer - worth another attempt."
        )
        concepts = question.get("concept_keys") or []
        if ordinal is None:
            continue
        moments.append(
            ReplayMoment(
                ordinal=int(ordinal),
                at_minute=round(at_minute, 2),
                prompt_text=str(question.get("prompt_text", "")),
                concept_key=concepts[0] if concepts else None,
                score=score,
                note=note,
                checkpoint_id=question.get("checkpoint_id"),
            )
        )
    return moments
