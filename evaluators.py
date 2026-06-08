"""Evaluator and self-improvement helpers for CareerPilot."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, field_validator

from agent import (
    CandidateProfile,
    ExtractedJobs,
    JobPosting,
    RankingResult,
    RankingRubric,
)
from prompts import evaluation_prompt, rubric_improvement_prompt


T = TypeVar("T", bound=BaseModel)


class StructuredGenerator(Protocol):
    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Generate a pydantic object from a prompt."""


class EvaluationReport(BaseModel):
    fit_quality: int = Field(ge=1, le=5)
    risk_detection: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    weaknesses: list[str] = Field(default_factory=list)
    explanation: str

    @field_validator("weaknesses", mode="before")
    @classmethod
    def normalize_weaknesses(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value


def evaluate_ranking(
    agent: StructuredGenerator,
    profile: CandidateProfile,
    jobs: list[JobPosting],
    ranking: RankingResult,
) -> EvaluationReport:
    """Ask Gemini to evaluate one ranking pass."""

    prompt = evaluation_prompt(
        profile_json=profile.model_dump_json(indent=2),
        jobs_json=ExtractedJobs(jobs=jobs).model_dump_json(indent=2),
        ranking_json=ranking.model_dump_json(indent=2),
    )
    return agent.generate_structured(prompt, EvaluationReport)


def needs_improvement(report: EvaluationReport, threshold: int = 4) -> bool:
    """Return true when any evaluator score is below the desired threshold."""

    return min(report.fit_quality, report.risk_detection, report.actionability) < threshold


def generate_improved_rubric(
    agent: StructuredGenerator,
    current_rubric: RankingRubric,
    evaluation: EvaluationReport,
    profile: CandidateProfile,
    jobs: list[JobPosting],
) -> RankingRubric:
    """Ask Gemini to produce a stronger rubric for a second ranking pass."""

    prompt = rubric_improvement_prompt(
        current_rubric=current_rubric,
        evaluation_json=evaluation.model_dump_json(indent=2),
        profile_json=profile.model_dump_json(indent=2),
        jobs=jobs,
    )
    improved = agent.generate_structured(prompt, RankingRubric)
    if improved.version == current_rubric.version:
        improved.version = "improved"
    return improved
