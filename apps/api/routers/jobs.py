"""Job description analysis."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import get_current_candidate, get_db, upload_rate_limit
from apps.api.schemas import JobAnalyzeRequest, JobAnalyzeResponse
from gauntlet.agents.intake import JobAnalyzerAgent
from gauntlet.content.companies import find_company
from gauntlet.content.taxonomy import display_name
from gauntlet.db.models import Candidate, Company, JobDescription
from gauntlet.services.documents import sanitise

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "/analyze",
    response_model=JobAnalyzeResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(upload_rate_limit)],
)
def analyze_job(
    payload: JobAnalyzeRequest,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> JobAnalyzeResponse:
    text = sanitise(payload.text)
    analysis = JobAnalyzerAgent().analyze(text)

    company_row: Company | None = None
    if payload.company:
        seed = find_company(payload.company)
        if seed is not None:
            company_row = session.scalar(select(Company).where(Company.slug == seed.slug))

    record = JobDescription(
        candidate_id=candidate.id,
        company_id=company_row.id if company_row else None,
        raw_text=text,
        title=payload.title or analysis.title,
        level=payload.level or analysis.level,
        analysis=analysis.model_dump(mode="json"),
    )
    session.add(record)
    session.flush()

    return JobAnalyzeResponse(
        id=record.id,
        title=record.title or analysis.title,
        level=record.level or analysis.level,
        must_have=analysis.must_have,
        weighted_concepts=[
            {
                "concept_key": item.concept_key,
                "display_name": display_name(item.concept_key),
                "weight": round(item.weight, 3),
            }
            for item in analysis.weighted_concepts
        ],
        summary=analysis.summary,
    )
