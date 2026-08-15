"""question-bank MCP server (spec section 16).

Exposes the question corpus over the Model Context Protocol, so an MCP-capable client
(Claude Desktop, an IDE, another agent) can search Gauntlet's questions without importing
Gauntlet.

Runs entirely from the in-process corpus, so it needs no database and no API key. That is
deliberate: a tool server that only works when the whole stack is up is a tool server
nobody runs.

**Security note that matters more than it looks.** Every tool here is read only, and none
of them accepts free text that becomes a query against anything but the local corpus.
Retrieved question text is *data*: it must never be fed back in as an instruction. That
rule lives at the boundary in ``gauntlet.agents.base``, and it is the reason this server
returns structured records rather than prose.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from gauntlet.content.companies import COMPANIES, find_company
from gauntlet.content.questions import QUESTIONS, questions_for_concepts
from gauntlet.content.taxonomy import concept_index, display_name
from gauntlet.evaluation.rubrics import get_rubric, rubric_index
from gauntlet.ingestion.dedup import QuestionCandidate, cluster_questions, find_duplicates_of
from gauntlet.retrieval.question_index import QuestionFilters, get_question_index
from gauntlet.schemas import InterviewType

server = MCPServer(
    name="gauntlet-question-bank",
    title="Gauntlet Question Bank",
    version="0.1.0",
    instructions=(
        "Search Gauntlet's technical interview question corpus. Every question is "
        "Gauntlet-authored and is never attributed to any company. Company interview "
        "mixes are archetype-based estimates, not observed reports, and must be "
        "presented as estimates."
    ),
)


def _as_record(seed: Any, score: float | None = None) -> dict[str, Any]:
    record = {
        "id": seed.slug,
        "question": seed.question,
        "interview_type": seed.interview_type.value,
        "concept_keys": list(seed.concept_keys),
        "concepts": [display_name(key) for key in seed.concept_keys],
        "topics": list(seed.topics),
        "difficulty": seed.difficulty,
        "rubric_key": seed.rubric_key,
        "expects_code": seed.expects_code,
        "follow_ups": list(seed.follow_ups),
        # Provenance travels with every question, always.
        "origin": "generated",
        "source_type": "gauntlet_authored",
        "attribution": "Gauntlet-authored. Not attributed to any company.",
    }
    if score is not None:
        record["relevance"] = round(score, 4)
    return record


@server.tool(
    description=(
        "Search interview questions by free text, with optional filters for interview "
        "type, concept and difficulty. Hybrid lexical and vector retrieval."
    )
)
def search_questions(
    query: str,
    interview_type: str | None = None,
    concept_key: str | None = None,
    min_difficulty: int = 1,
    max_difficulty: int = 5,
    limit: int = 10,
) -> dict[str, Any]:
    """Find questions matching a query."""
    try:
        types = frozenset({InterviewType(interview_type)}) if interview_type else None
    except ValueError:
        return {
            "error": f"Unknown interview_type '{interview_type}'.",
            "valid": [item.value for item in InterviewType],
        }

    filters = QuestionFilters(
        interview_types=types,
        concept_keys=frozenset({concept_key}) if concept_key else None,
        min_difficulty=max(1, min_difficulty),
        max_difficulty=min(5, max_difficulty),
    )
    results = get_question_index().search(query, filters, limit=max(1, min(limit, 50)))
    return {
        "query": query,
        "count": len(results),
        "questions": [_as_record(item.seed, item.score) for item in results],
    }


@server.tool(
    description=(
        "Find questions that are the same underlying problem as the given text, even "
        "when worded completely differently. Use before adding a question to avoid "
        "duplicating one already in the bank."
    )
)
def find_question_family(question: str, concept_keys: list[str] | None = None) -> dict[str, Any]:
    """Check a question against the corpus for existing variants."""
    corpus = [
        QuestionCandidate(
            id=seed.slug,
            text=seed.question,
            concept_keys=tuple(seed.concept_keys),
            topics=tuple(seed.topics),
        )
        for seed in QUESTIONS
    ]
    query = QuestionCandidate(
        id="__query__", text=question, concept_keys=tuple(concept_keys or ())
    )
    matches = find_duplicates_of(query, corpus)
    return {
        "question": question,
        "is_duplicate": bool(matches),
        "matches": [
            {
                "id": candidate.id,
                "question": candidate.text,
                "similarity": score.combined,
                "explanation": score.explain(),
            }
            for candidate, score in matches
        ],
    }


@server.tool(description="List every question tagged to a concept, hardest first.")
def get_topic_questions(concept_key: str, limit: int = 20) -> dict[str, Any]:
    """Questions for one concept."""
    if concept_key not in concept_index():
        return {"error": f"Unknown concept '{concept_key}'.", "hint": "Call list_concepts."}

    seeds = questions_for_concepts({concept_key})
    seeds.sort(key=lambda seed: seed.difficulty, reverse=True)
    return {
        "concept_key": concept_key,
        "display_name": display_name(concept_key),
        "count": len(seeds),
        "questions": [_as_record(seed) for seed in seeds[:limit]],
    }


@server.tool(
    description=(
        "Estimated interview shape for a company. Always an estimate derived from the "
        "general shape of that kind of engineering organisation, never observed reports."
    )
)
def get_company_patterns(company: str) -> dict[str, Any]:
    """Interview mix for a company, clearly labelled as an estimate."""
    found = find_company(company)
    if found is None:
        return {
            "error": f"Unknown company '{company}'.",
            "known": [item.slug for item in COMPANIES][:60],
        }
    mix = found.interview_mix()
    return {
        "slug": found.slug,
        "name": found.name,
        "sector": found.sector,
        "evidence": mix["evidence"],
        "basis": mix["basis"],
        "disclaimer": mix["disclaimer"],
        "distribution": mix["distribution"],
    }


@server.tool(
    description=(
        "Full metadata for one question, including the grading rubric used to score "
        "answers to it and the known misconceptions for that concept."
    )
)
def get_question_metadata(question_id: str) -> dict[str, Any]:
    """Everything known about one question."""
    seed = next((item for item in QUESTIONS if item.slug == question_id), None)
    if seed is None:
        return {"error": f"Unknown question id '{question_id}'."}

    rubric = get_rubric(seed.rubric_key, seed.interview_type)
    return {
        **_as_record(seed),
        "rubric": {
            "key": rubric.key,
            "title": rubric.title,
            "dimensions": [
                {"key": item.key, "label": item.label, "probe": item.hint}
                for item in rubric.dimensions
            ],
            "known_misconceptions": [
                {"belief": item.belief, "correction": item.correction}
                for item in rubric.common_misconceptions
            ],
        },
    }


@server.tool(description="List the concept taxonomy, optionally filtered by interview type.")
def list_concepts(interview_type: str | None = None) -> dict[str, Any]:
    """The concept keys questions are tagged with."""
    concepts = list(concept_index().values())
    if interview_type:
        try:
            wanted = InterviewType(interview_type)
        except ValueError:
            return {"error": f"Unknown interview_type '{interview_type}'."}
        concepts = [item for item in concepts if item.interview_type is wanted]
    return {
        "count": len(concepts),
        "concepts": [
            {
                "key": item.key,
                "display_name": item.display_name,
                "interview_type": item.interview_type.value,
                "difficulty_range": [item.difficulty_floor, item.difficulty_ceiling],
            }
            for item in concepts
        ],
    }


@server.resource(
    "gauntlet://corpus/stats",
    description="Corpus size, family count and duplication ratio.",
    mime_type="application/json",
)
def corpus_stats() -> dict[str, Any]:
    """Health of the question bank, including how much duplication it contains."""
    from gauntlet.ingestion.dedup import duplication_report

    candidates = [
        QuestionCandidate(
            id=seed.slug,
            text=seed.question,
            concept_keys=tuple(seed.concept_keys),
            topics=tuple(seed.topics),
        )
        for seed in QUESTIONS
    ]
    report = duplication_report(cluster_questions(candidates))
    return {
        **report,
        "concepts": len(concept_index()),
        "rubrics": len(rubric_index()),
        "companies": len(COMPANIES),
        "attribution": "All questions are Gauntlet-authored. None are attributed to a company.",
    }


def main() -> None:
    """Run over stdio, which is how MCP clients launch a local server."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
