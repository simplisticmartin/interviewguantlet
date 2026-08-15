"""candidate MCP server (spec section 16).

Exposes a candidate's own history: resume, past interviews, skill graph, and the
misconceptions they are still carrying. This is what lets an MCP-capable client answer
"what should I study tonight?" using real measured evidence rather than a guess.

**This server reads personal data, so two constraints are structural.**

Every tool takes an explicit ``candidate_id`` and returns data for that candidate only.
There is no "current user" and no ambient scope, because ambient scope is how a tool
server ends up handing one person's interview history to another.

And it runs over stdio, meaning it is launched as a local subprocess by the client and
inherits that user's trust. That is fine locally and is *not* fine over the network: a
remote deployment needs real authentication and per-request authorisation before this is
exposed, which is noted here rather than left for someone to discover.

It requires the database. When Postgres is unreachable, tools return a clear error rather
than an empty result, because an empty skill graph and an unavailable one mean very
different things to whatever is reading them.
"""

from __future__ import annotations

import uuid
from typing import Any

from mcp.server import MCPServer
from sqlalchemy import select

from gauntlet.content.taxonomy import display_name
from gauntlet.db.models import Candidate, InterviewSession, Misconception, Resume
from gauntlet.db.session import database_available, session_scope
from gauntlet.services.skills import due_for_review, load_readings
from gauntlet.skills.mastery import classify_calibration

server = MCPServer(
    name="gauntlet-candidate",
    title="Gauntlet Candidate History",
    version="0.1.0",
    instructions=(
        "Read a candidate's interview history, skill graph and outstanding "
        "misconceptions. Every tool requires an explicit candidate_id. Scores are "
        "measured from answers actually given; areas never assessed are reported as "
        "untested rather than as zero."
    ),
)

_DB_ERROR = {
    "error": "Database unavailable.",
    "hint": "Start it with: docker compose up -d db redis && alembic upgrade head",
}


def _resolve(session: Any, candidate_id: str) -> Candidate | None:
    try:
        parsed = uuid.UUID(candidate_id)
    except ValueError:
        return None
    return session.get(Candidate, parsed)


@server.tool(description="The candidate's most recent resume, parsed into structured claims.")
def get_resume(candidate_id: str) -> dict[str, Any]:
    """Parsed resume profile and the claims worth cross-examining."""
    if not database_available():
        return _DB_ERROR
    with session_scope() as session:
        candidate = _resolve(session, candidate_id)
        if candidate is None:
            return {"error": f"Unknown candidate '{candidate_id}'."}
        resume = session.scalar(
            select(Resume)
            .where(Resume.candidate_id == candidate.id)
            .order_by(Resume.created_at.desc())
        )
        if resume is None:
            return {"candidate_id": candidate_id, "resume": None}
        profile = resume.profile or {}
        return {
            "candidate_id": candidate_id,
            "filename": resume.filename,
            "years_experience": profile.get("years_experience"),
            "primary_languages": profile.get("primary_languages", []),
            "frameworks": profile.get("frameworks", []),
            "claims": profile.get("claims", []),
        }


@server.tool(description="Past interviews with their scores and outcomes, most recent first.")
def get_previous_interviews(candidate_id: str, limit: int = 10) -> dict[str, Any]:
    """Interview history."""
    if not database_available():
        return _DB_ERROR
    with session_scope() as session:
        candidate = _resolve(session, candidate_id)
        if candidate is None:
            return {"error": f"Unknown candidate '{candidate_id}'."}
        rows = session.scalars(
            select(InterviewSession)
            .where(InterviewSession.candidate_id == candidate.id)
            .order_by(InterviewSession.created_at.desc())
            .limit(max(1, min(limit, 50)))
        ).all()
        return {
            "candidate_id": candidate_id,
            "count": len(rows),
            "interviews": [
                {
                    "id": str(row.id),
                    "role": row.target_role,
                    "level": row.target_level,
                    "mode": row.mode,
                    "status": row.status,
                    "questions_asked": len(row.questions),
                    "overall": (row.final_scorecard or {}).get("overall"),
                    "recommendation": (row.final_scorecard or {}).get("committee", {}).get(
                        "recommendation"
                    ),
                    "ended_at": row.ended_at.isoformat() if row.ended_at else None,
                }
                for row in rows
            ],
        }


@server.tool(
    description=(
        "The candidate's skill graph: measured mastery per concept, our confidence in "
        "that measurement, and the calibration quadrant."
    )
)
def get_mastery_graph(candidate_id: str) -> dict[str, Any]:
    """Current skill readings."""
    if not database_available():
        return _DB_ERROR
    with session_scope() as session:
        candidate = _resolve(session, candidate_id)
        if candidate is None:
            return {"error": f"Unknown candidate '{candidate_id}'."}
        readings = load_readings(session, candidate.id)
        return {
            "candidate_id": candidate_id,
            "count": len(readings),
            "note": "Only concepts actually assessed appear here. Absence means untested.",
            "skills": [
                {
                    "concept_key": reading.concept_key,
                    "display_name": reading.display_name,
                    "mastery": round(reading.mastery, 3),
                    "confidence": round(reading.confidence, 3),
                    "evidence_count": reading.evidence_count,
                    "calibration": classify_calibration(
                        reading.mastery, reading.self_confidence
                    ).value,
                }
                for reading in readings
            ],
        }


@server.tool(
    description=(
        "Things the candidate believes that are wrong, with the correction. These are "
        "the highest value study targets because nobody revises what they think they "
        "already know."
    )
)
def get_previous_mistakes(candidate_id: str) -> dict[str, Any]:
    """Open misconceptions, most severe first."""
    if not database_available():
        return _DB_ERROR
    with session_scope() as session:
        candidate = _resolve(session, candidate_id)
        if candidate is None:
            return {"error": f"Unknown candidate '{candidate_id}'."}
        rows = session.scalars(
            select(Misconception)
            .where(
                Misconception.candidate_id == candidate.id,
                Misconception.status == "open",
            )
            .order_by(Misconception.severity.desc(), Misconception.times_observed.desc())
        ).all()
        return {
            "candidate_id": candidate_id,
            "count": len(rows),
            "misconceptions": [
                {
                    "concept_key": row.concept_key,
                    "display_name": display_name(row.concept_key),
                    "belief": row.belief,
                    "correction": row.correction,
                    "severity": row.severity,
                    "times_observed": row.times_observed,
                }
                for row in rows
            ],
        }


@server.tool(
    description=(
        "Concepts due for review under spaced repetition, weakest first. Use this to "
        "answer 'what should I revise now?'."
    )
)
def get_skill_history(candidate_id: str, limit: int = 10) -> dict[str, Any]:
    """What is due for review."""
    if not database_available():
        return _DB_ERROR
    with session_scope() as session:
        candidate = _resolve(session, candidate_id)
        if candidate is None:
            return {"error": f"Unknown candidate '{candidate_id}'."}
        due = due_for_review(session, candidate.id, limit=max(1, min(limit, 50)))
        return {
            "candidate_id": candidate_id,
            "due_count": len(due),
            "due_for_review": [
                {
                    "concept_key": reading.concept_key,
                    "display_name": reading.display_name,
                    "mastery": round(reading.mastery, 3),
                    "evidence_count": reading.evidence_count,
                }
                for reading in due
            ],
        }


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
