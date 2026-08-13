"""Resume upload, parsing, and retrieval.

Uploaded bytes are untrusted. They are size-capped, content-type checked, text-extracted,
and stored as text - the original binary is never re-served, and nothing in the document
is ever treated as an instruction.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.deps import get_current_candidate, get_db, upload_rate_limit
from apps.api.schemas import ResumeDetail, ResumeSummary, ResumeTextRequest
from gauntlet.agents.intake import ResumeParserAgent
from gauntlet.config import get_settings
from gauntlet.db.models import Candidate, Resume, ResumeClaim
from gauntlet.services.documents import DocumentError, extract_text, sanitise

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/resumes", tags=["resumes"])


@router.post(
    "",
    response_model=ResumeDetail,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(upload_rate_limit)],
)
async def upload_resume(
    file: UploadFile = File(...),
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> ResumeDetail:
    settings = get_settings()
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_bytes // 1024 // 1024} MB.",
        )

    try:
        text = extract_text(data, file.filename or "resume", file.content_type)
    except DocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return _store(session, candidate, file.filename or "resume", text, file.content_type)


@router.post(
    "/text",
    response_model=ResumeDetail,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(upload_rate_limit)],
)
def paste_resume(
    payload: ResumeTextRequest,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> ResumeDetail:
    return _store(session, candidate, payload.filename, sanitise(payload.text), "text/plain")


@router.get("", response_model=list[ResumeSummary])
def list_resumes(
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> list[ResumeSummary]:
    rows = session.scalars(
        select(Resume)
        .where(Resume.candidate_id == candidate.id)
        .order_by(Resume.created_at.desc())
    ).all()
    return [_summary(row) for row in rows]


@router.get("/{resume_id}", response_model=ResumeDetail)
def get_resume(
    resume_id: uuid.UUID,
    candidate: Candidate = Depends(get_current_candidate),
    session: Session = Depends(get_db),
) -> ResumeDetail:
    row = session.get(Resume, resume_id)
    if row is None or row.candidate_id != candidate.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resume not found.")
    return _detail(row)


def _store(
    session: Session,
    candidate: Candidate,
    filename: str,
    text: str,
    content_type: str | None,
) -> ResumeDetail:
    profile = ResumeParserAgent().parse(text)

    # A newly uploaded resume becomes the primary one.
    for previous in session.scalars(
        select(Resume).where(Resume.candidate_id == candidate.id, Resume.is_primary.is_(True))
    ):
        previous.is_primary = False

    resume = Resume(
        candidate_id=candidate.id,
        filename=filename[:400],
        content_type=(content_type or "text/plain")[:160],
        raw_text=text,
        profile=profile.model_dump(mode="json"),
        is_primary=True,
    )
    session.add(resume)
    session.flush()

    for claim in profile.claims:
        session.add(
            ResumeClaim(
                resume_id=resume.id,
                claim_text=claim.claim_text,
                claim_type=claim.claim_type,
                technologies=claim.technologies,
                concept_keys=claim.concept_keys,
                has_metric=claim.has_metric,
                probe_priority=claim.probe_priority,
            )
        )

    if profile.display_name and profile.display_name != "Candidate":
        candidate.display_name = profile.display_name

    log.info(
        "resume.stored",
        candidate=str(candidate.id),
        concepts=len(profile.concept_keys),
        claims=len(profile.claims),
    )
    session.flush()
    return _detail(resume)


def _summary(row: Resume) -> ResumeSummary:
    profile = row.profile or {}
    return ResumeSummary(
        id=row.id,
        filename=row.filename,
        created_at=row.created_at,
        is_primary=row.is_primary,
        years_experience=float(profile.get("years_experience", 0.0)),
        concept_count=len(profile.get("concept_keys", [])),
        claim_count=len(profile.get("claims", [])),
    )


def _detail(row: Resume) -> ResumeDetail:
    summary = _summary(row)
    return ResumeDetail(
        **summary.model_dump(),
        profile=row.profile or {},
        excerpt=row.raw_text[:1200],
    )
