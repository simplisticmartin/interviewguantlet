"""Question corpus search and company intelligence (spec sections 11, 26, 36).

Question search runs in Postgres - trigram + metadata filters + optional vector
similarity - because the corpus is meant to grow far beyond what belongs in memory.
The in-process index in ``gauntlet.retrieval`` serves the live interview instead, where
latency matters more than corpus size.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import Select, Text, cast, func, or_, select
from sqlalchemy.orm import Session

from apps.api.deps import get_current_candidate, get_db
from apps.api.schemas import CompanyPatterns, CompanyView, QuestionSearchResult
from gauntlet.content.companies import COMPANIES, find_company
from gauntlet.content.taxonomy import display_name
from gauntlet.db.models import Candidate, Question
from gauntlet.llm.embeddings import get_embedder
from gauntlet.services.skills import load_readings

router = APIRouter(tags=["catalog"])


@router.get("/questions/search", response_model=list[QuestionSearchResult])
def search_questions(
    q: Annotated[str, Query(max_length=300)] = "",
    interview_type: Annotated[str | None, Query(max_length=60)] = None,
    concept: Annotated[str | None, Query(max_length=200)] = None,
    min_difficulty: Annotated[int, Query(ge=1, le=5)] = 1,
    max_difficulty: Annotated[int, Query(ge=1, le=5)] = 5,
    origin: Annotated[str | None, Query(max_length=40)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    semantic: bool = True,
    session: Session = Depends(get_db),
    _: Candidate = Depends(get_current_candidate),
) -> list[QuestionSearchResult]:
    stmt: Select[Any] = select(Question).where(
        Question.is_active.is_(True),
        Question.difficulty.between(min_difficulty, max_difficulty),
    )
    if interview_type:
        stmt = stmt.where(Question.interview_type == interview_type)
    if origin:
        stmt = stmt.where(Question.question_origin == origin)
    if concept:
        stmt = stmt.where(Question.concept_keys.contains([concept]))

    if q.strip():
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(
            or_(
                Question.question.ilike(pattern),
                cast(Question.topics, Text).ilike(pattern),
            )
        )

    rows = list(session.scalars(stmt.limit(200)))

    # Vector rerank over the filtered set. Metadata filters run first so the vector
    # search never has to consider the whole corpus (spec section 12).
    if semantic and q.strip() and rows:
        embedder = get_embedder()
        query_vector = embedder.embed_one(q)
        from gauntlet.llm.embeddings import cosine_similarity

        scored = [
            (
                row,
                cosine_similarity(query_vector, list(row.embedding))
                if row.embedding is not None
                else 0.0,
            )
            for row in rows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [_to_result(row, score) for row, score in scored[:limit]]

    return [_to_result(row, None) for row in rows[:limit]]


@router.get("/companies", response_model=list[CompanyView])
def list_companies(
    sector: Annotated[str | None, Query(max_length=60)] = None,
) -> list[CompanyView]:
    return [
        CompanyView(
            slug=company.slug,
            name=company.name,
            sector=company.sector,
            aliases=list(company.aliases),
        )
        for company in COMPANIES
        if sector is None or company.sector == sector
    ]


@router.get("/companies/{slug}/patterns", response_model=CompanyPatterns)
def company_patterns(
    slug: str,
    level: Annotated[str, Query(max_length=60)] = "senior",
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> CompanyPatterns:
    company = find_company(slug)
    if company is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown company.")

    mix = company.interview_mix()
    raw_distribution = mix["distribution"]
    if not isinstance(raw_distribution, dict):  # pragma: no cover - defensive
        raw_distribution = {}
    distribution = {str(k): float(v) for k, v in raw_distribution.items()}

    readings = load_readings(session, candidate.id)
    readiness = _readiness(distribution, readings, level) if readings else None

    return CompanyPatterns(
        slug=company.slug,
        name=company.name,
        sector=company.sector,
        evidence=str(mix["evidence"]),
        basis=str(mix["basis"]),
        disclaimer=str(mix["disclaimer"]),
        distribution=distribution,
        readiness=readiness,
    )


def _readiness(
    distribution: dict[str, float], readings: list[Any], level: str
) -> dict[str, Any]:
    """Compare measured mastery to the estimated interview mix (spec section 36).

    Explicitly an estimate of *preparedness*, never a hiring probability. Areas with no
    evidence are reported as unmeasured rather than scored as zero - claiming someone is
    0% ready at something never asked about would be a lie.
    """
    from gauntlet.content.taxonomy import get_concept

    by_type: dict[str, list[Any]] = {}
    for reading in readings:
        if not reading.evidence_count:
            continue
        concept = get_concept(reading.concept_key)
        if concept is None:
            continue
        by_type.setdefault(concept.interview_type.value, []).append(reading)

    areas: list[dict[str, Any]] = []
    covered_weight = 0.0
    weighted_score = 0.0

    for interview_type, weight in sorted(
        distribution.items(), key=lambda item: item[1], reverse=True
    ):
        group = by_type.get(interview_type, [])
        if group:
            mastery = sum(item.mastery for item in group) / len(group)
            covered_weight += weight
            weighted_score += weight * mastery
            areas.append(
                {
                    "interview_type": interview_type,
                    "weight": round(weight, 3),
                    "score": round(100 * mastery),
                    "evidence_count": sum(item.evidence_count for item in group),
                    "measured": True,
                }
            )
        else:
            areas.append(
                {
                    "interview_type": interview_type,
                    "weight": round(weight, 3),
                    "score": None,
                    "evidence_count": 0,
                    "measured": False,
                }
            )

    estimated = round(100 * weighted_score / covered_weight) if covered_weight else None
    return {
        "level": level,
        "estimated_readiness": estimated,
        "coverage": round(covered_weight, 3),
        "areas": areas,
        "caveat": (
            "An estimate of preparedness against Gauntlet's estimated interview mix, "
            "based only on areas you have actually been assessed on. Not a prediction "
            "of any hiring outcome."
        ),
    }


def _to_result(row: Question, score: float | None) -> QuestionSearchResult:
    return QuestionSearchResult(
        id=row.id,
        question=row.question,
        interview_type=row.interview_type,
        concept_keys=[display_name(key) for key in row.concept_keys],
        topics=list(row.topics),
        difficulty=row.difficulty,
        rubric_key=row.rubric_key,
        question_origin=row.question_origin,
        source_type=row.source_type,
        score=round(score, 4) if score is not None else None,
    )


@router.get("/questions/count")
def question_count(session: Session = Depends(get_db)) -> dict[str, int]:
    total = session.scalar(select(func.count()).select_from(Question)) or 0
    return {"total": int(total)}
