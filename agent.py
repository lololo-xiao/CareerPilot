"""Gemini-powered CareerPilot agent logic."""

from __future__ import annotations

import os
from typing import Callable, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, field_validator


T = TypeVar("T", bound=BaseModel)


class LanguageProfile(BaseModel):
    language: str = Field(description="Language name, e.g. English or German.")
    level: str = Field(description="Observed or inferred level, e.g. B2, fluent.")
    evidence: str = Field(default="", description="Short evidence or uncertainty.")


class ControlledProfileMetric(BaseModel):
    key: str = Field(description="Field key from profile.json.")
    label: str = Field(default="", description="Human-readable field label.")
    value: str = Field(
        default="",
        description="Extracted value. Use compact JSON text when the value is structured.",
    )
    evidence: list[str] = Field(default_factory=list)
    confidence: str = Field(default="", description="High, medium, low, or unknown.")
    uncertainty: str = Field(default="", description="What is missing, inferred, or conflicting.")


class CandidateProfile(BaseModel):
    target_roles: list[str] = Field(default_factory=list)
    core_skills: list[str] = Field(default_factory=list)
    secondary_skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    education: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    languages: list[LanguageProfile] = Field(default_factory=list)
    visa_status: str = ""
    blue_card_relevance: str = ""
    seniority_level: str = ""
    seniority_target: str = ""
    location_flexibility: str = ""
    notable_strengths: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    uncertainty_fields: list[str] = Field(default_factory=list)
    controlled_metrics: list[ControlledProfileMetric] = Field(default_factory=list)


class JobPosting(BaseModel):
    job_index: int = Field(ge=1)
    title: str
    company: str = ""
    location: str = ""
    remote_policy: str = ""
    seniority: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_years_experience: float | None = None
    language_requirements: list[str] = Field(default_factory=list)
    visa_blue_card_signals: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    risks_for_international_graduate: list[str] = Field(default_factory=list)
    raw_summary: str = ""


class ExtractedJobs(BaseModel):
    jobs: list[JobPosting] = Field(default_factory=list)


class RubricWeights(BaseModel):
    target_role_fit: int = 20
    skill_match: int = 25
    experience_match: int = 15
    location_fit: int = 10
    visa_blue_card_friendliness: int = 15
    language_requirement: int = 10
    seniority_risk: int = 5

    @field_validator("*")
    @classmethod
    def weight_must_be_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Rubric weights must be non-negative.")
        return value


class RankingRubric(BaseModel):
    version: str = "baseline"
    weights: RubricWeights = Field(default_factory=RubricWeights)
    guidance: list[str] = Field(
        default_factory=lambda: [
            "Prioritize jobs that match the candidate's stated target roles.",
            "Treat German language requirements as a major risk unless the CV shows matching German ability.",
            "Reward explicit visa sponsorship, relocation support, EU Blue Card eligibility, or salary transparency.",
            "Penalize senior roles if the CV suggests graduate, junior, or early-career experience.",
            "Prefer Germany-based or Germany-remote jobs when the candidate targets Germany.",
        ]
    )


class DimensionScores(BaseModel):
    target_role_fit: int = Field(ge=0, le=100)
    skill_match: int = Field(ge=0, le=100)
    experience_match: int = Field(ge=0, le=100)
    location_fit: int = Field(ge=0, le=100)
    visa_blue_card_friendliness: int = Field(ge=0, le=100)
    language_requirement: int = Field(ge=0, le=100)
    seniority_risk: int = Field(ge=0, le=100)


class JobMatch(BaseModel):
    job_index: int = Field(ge=1)
    title: str
    company: str = ""
    location: str = ""
    match_score: int = Field(ge=0, le=100)
    dimension_scores: DimensionScores
    matched_evidence: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    realistic_assessment: str = ""
    next_action: str


class RankingResult(BaseModel):
    rubric_version: str
    rubric_notes: list[str] = Field(default_factory=list)
    matches: list[JobMatch] = Field(default_factory=list)


class CareerPilotAgent:
    """Small wrapper around Gemini structured-output calls."""

    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = self._build_client()

    def extract_profile(
        self,
        cv_text: str,
        profile_definition_json: str = "{}",
    ) -> CandidateProfile:
        from prompts import CV_EXTRACTION_PROMPT

        return self.generate_structured(
            CV_EXTRACTION_PROMPT.format(
                profile_definition_json=profile_definition_json.strip() or "{}",
                cv_text=cv_text.strip(),
            ),
            CandidateProfile,
        )

    def extract_jobs(self, job_text: str) -> list[JobPosting]:
        from prompts import JOB_EXTRACTION_PROMPT

        extracted = self.generate_structured(
            JOB_EXTRACTION_PROMPT.format(job_text=job_text.strip()),
            ExtractedJobs,
        )
        return extracted.jobs

    def rank_jobs(
        self,
        profile: CandidateProfile,
        jobs: list[JobPosting],
        rubric: RankingRubric,
        top_n: int = 5,
    ) -> RankingResult:
        from prompts import ranking_prompt

        prompt = ranking_prompt(
            profile_json=profile.model_dump_json(indent=2),
            jobs_json=ExtractedJobs(jobs=jobs).model_dump_json(indent=2),
            rubric=rubric,
            top_n=top_n,
        )
        result = self.generate_structured(prompt, RankingResult)
        result.rubric_version = rubric.version
        return result

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty structured response.")
        return schema.model_validate_json(response.text)

    def generate_structured_with_retry(
        self,
        prompt: str,
        schema: type[T],
        attempts: int = 2,
        on_retry: "Callable[[int, str], None] | None" = None,
    ) -> T:
        """Self-correcting structured generation.

        On an empty response or a Pydantic validation error, the agent re-prompts
        itself once with the failure appended so the model can repair its own
        output. This is the autonomous "retry after parsing failure" behaviour.
        """

        last_error: Exception | None = None
        current_prompt = prompt
        for attempt in range(1, attempts + 1):
            try:
                return self.generate_structured(current_prompt, schema)
            except (ValidationError, RuntimeError, ValueError) as exc:
                last_error = exc
                if on_retry is not None:
                    on_retry(attempt, str(exc))
                if attempt < attempts:
                    current_prompt = (
                        f"{prompt}\n\nYour previous answer failed schema validation "
                        f"with this error:\n{exc}\n\nReturn corrected JSON that "
                        f"strictly matches the required schema."
                    )
        raise RuntimeError(
            f"Structured generation failed after {attempts} attempts: {last_error}"
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _build_client() -> genai.Client:
        use_vertex = _as_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if use_vertex or project:
            if not project:
                raise RuntimeError(
                    "Set GOOGLE_CLOUD_PROJECT when GOOGLE_GENAI_USE_VERTEXAI=true."
                )
            return genai.Client(vertexai=True, project=project, location=location)

        if api_key:
            return genai.Client(api_key=api_key)

        return genai.Client()


def default_rubric() -> RankingRubric:
    return RankingRubric()


def _as_bool(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}
