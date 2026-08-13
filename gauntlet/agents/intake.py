"""Intake agents: resume parsing and job-description analysis.

Both operate on wholly untrusted uploaded text, so both go through the fenced-block
path in :class:`Agent` and neither is ever allowed to influence tool selection.
"""

from __future__ import annotations

from gauntlet.agents.base import Agent
from gauntlet.content.taxonomy import taxonomy_for_prompt
from gauntlet.prompts.catalog import JOB_ANALYZER, RESUME_PARSER
from gauntlet.schemas import JobAnalysis, ResumeProfile


class ResumeParserAgent(Agent):
    key = "resume_parser"

    def parse(self, raw_text: str) -> ResumeProfile:
        result = self.invoke(
            RESUME_PARSER,
            ResumeProfile,
            context={
                "taxonomy": taxonomy_for_prompt(),
                "instruction": (
                    "Extract the profile and the specific claims worth cross-examining."
                ),
            },
            blocks={"resume": raw_text},
        )
        return _sanitise_profile(result.value)


class JobAnalyzerAgent(Agent):
    key = "job_analyzer"

    def analyze(self, raw_text: str) -> JobAnalysis:
        result = self.invoke(
            JOB_ANALYZER,
            JobAnalysis,
            context={
                "taxonomy": taxonomy_for_prompt(),
                "instruction": "Weight concepts by interview-assessment likelihood.",
            },
            blocks={"job_description": raw_text},
        )
        return _sanitise_job(result.value)


def _known(keys: list[str]) -> list[str]:
    """Drop concept keys the model invented - the taxonomy is the source of truth."""
    from gauntlet.content.taxonomy import concept_index

    index = concept_index()
    return [key for key in keys if key in index]


def _sanitise_profile(profile: ResumeProfile) -> ResumeProfile:
    return profile.model_copy(
        update={
            "concept_keys": _known(profile.concept_keys),
            "claims": [
                claim.model_copy(update={"concept_keys": _known(claim.concept_keys)})
                for claim in profile.claims
            ],
        }
    )


def _sanitise_job(job: JobAnalysis) -> JobAnalysis:
    from gauntlet.content.taxonomy import concept_index

    index = concept_index()
    return job.model_copy(
        update={
            "weighted_concepts": [
                concept for concept in job.weighted_concepts if concept.concept_key in index
            ]
        }
    )
