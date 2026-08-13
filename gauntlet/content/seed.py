"""Seed the database from the in-code content catalogues.

Idempotent: every row is upserted on its natural key, so running this repeatedly is
safe and re-running after editing a rubric or adding a company updates in place.

The code catalogues remain the source of truth. The database copy exists so that
corpus browsing, question search, and company analytics can be served with SQL at
scale, and so that user-contributed content (roadmap phase 10) has somewhere to live
alongside the shipped content with its provenance intact.
"""

from __future__ import annotations

import argparse

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from gauntlet.content.companies import COMPANIES
from gauntlet.content.questions import QUESTIONS
from gauntlet.content.taxonomy import CONCEPTS
from gauntlet.db.models import Company, Concept, PromptVersion, Question, Rubric
from gauntlet.db.session import session_scope
from gauntlet.evaluation.rubrics import RUBRICS
from gauntlet.llm.embeddings import get_embedder
from gauntlet.prompts.catalog import ALL_PROMPTS

log = structlog.get_logger(__name__)


def seed_concepts(session: Session) -> int:
    existing = {row.key: row for row in session.scalars(select(Concept))}
    touched = 0
    for concept in CONCEPTS:
        row = existing.get(concept.key)
        if row is None:
            row = Concept(key=concept.key)
            session.add(row)
        row.parent_key = concept.parent_key
        row.display_name = concept.display_name
        row.domain = concept.domain
        row.interview_type = concept.interview_type.value
        row.difficulty_floor = concept.difficulty_floor
        row.difficulty_ceiling = concept.difficulty_ceiling
        row.aliases = list(concept.aliases)
        touched += 1
    return touched


def seed_companies(session: Session) -> int:
    existing = {row.slug: row for row in session.scalars(select(Company))}
    touched = 0
    for company in COMPANIES:
        row = existing.get(company.slug)
        if row is None:
            row = Company(slug=company.slug)
            session.add(row)
        row.name = company.name
        row.sector = company.sector
        row.aliases = list(company.aliases)
        row.interview_mix = company.interview_mix()
        row.notes = (
            "Interview mix is an archetype-based estimate. Gauntlet holds no observed "
            "interview reports for this company."
        )
        touched += 1
    return touched


def seed_rubrics(session: Session) -> int:
    existing = {(row.key, row.version): row for row in session.scalars(select(Rubric))}
    touched = 0
    for rubric in RUBRICS:
        row = existing.get((rubric.key, rubric.version))
        if row is None:
            row = Rubric(key=rubric.key, version=rubric.version)
            session.add(row)
        row.title = rubric.title
        row.concept_key = rubric.concept_key
        row.dimensions = [dimension.model_dump() for dimension in rubric.dimensions]
        touched += 1
    return touched


def seed_prompts(session: Session) -> int:
    """Mirror the prompt catalogue so evaluations can reference a concrete row."""
    existing = {(row.name, row.version): row for row in session.scalars(select(PromptVersion))}
    touched = 0
    for template in ALL_PROMPTS:
        row = existing.get((template.name, template.version))
        if row is None:
            row = PromptVersion(name=template.name, version=template.version)
            session.add(row)
        row.template = f"SYSTEM:\n{template.system}\n\nUSER:\n{template.user}"
        row.temperature = template.temperature
        row.checksum = template.checksum
        touched += 1
    return touched


def seed_questions(session: Session, embed: bool = True) -> int:
    """Load the authored corpus, embedding question text for hybrid retrieval."""
    existing = {row.question: row for row in session.scalars(select(Question))}
    embedder = get_embedder() if embed else None

    to_embed: list[tuple[Question, str]] = []
    touched = 0

    for seed in QUESTIONS:
        row = existing.get(seed.question)
        if row is None:
            row = Question(question=seed.question)
            session.add(row)
        row.follow_ups = list(seed.follow_ups)
        row.role_family = seed.role_family
        row.level = seed.level
        row.interview_type = seed.interview_type.value
        row.concept_keys = list(seed.concept_keys)
        row.topics = list(seed.topics)
        row.difficulty = seed.difficulty
        row.rubric_key = seed.rubric_key
        # Everything shipped is Gauntlet-authored. Nothing here is attributed to a
        # company, and nothing was scraped (spec sections 7 and 13).
        row.question_origin = "generated"
        row.source_type = "gauntlet_authored"
        row.copyright_status = "original"
        row.confidence = 1.0
        row.based_on_patterns = list(seed.based_on_patterns)
        row.is_active = True
        touched += 1
        if embedder is not None:
            to_embed.append((row, f"{seed.question} {' '.join(seed.topics)}"))

    if embedder is not None and to_embed:
        vectors = embedder.embed([text for _, text in to_embed])
        for (row, _), vector in zip(to_embed, vectors, strict=True):
            row.embedding = vector

    return touched


def seed_all(session: Session, embed: bool = True) -> dict[str, int]:
    counts = {
        "concepts": seed_concepts(session),
        "companies": seed_companies(session),
        "rubrics": seed_rubrics(session),
        "prompts": seed_prompts(session),
        "questions": seed_questions(session, embed=embed),
    }
    session.flush()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the Gauntlet database.")
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding generation (faster; question vector search will be empty).",
    )
    args = parser.parse_args()

    with session_scope() as session:
        counts = seed_all(session, embed=not args.no_embed)

    for name, count in counts.items():
        print(f"  {name:<12} {count}")
    print("Seed complete.")


if __name__ == "__main__":
    main()
