"""Prompt builders for CareerPilot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import JobPosting, RankingRubric


CV_EXTRACTION_PROMPT = """\
You are CareerPilot, a job-search agent for international graduates in Germany.

Extract a structured candidate profile from the candidate input. The input may
include CV text extracted from an uploaded PDF, fallback pasted CV text, and
additional notes from the user. Focus on facts and stated preferences that are
useful for ranking German jobs: target roles, skills, years of experience,
education, German/EU location preferences, languages, work authorization, visa
or Blue Card constraints, personal constraints, and seniority level.

Treat the user's additional notes as explicit context for preferences,
constraints, target roles, visa needs, location preferences, and personal
context that may not appear in the CV. If CV facts and user notes conflict, keep
the conflict visible in constraints or uncertainties. If a field is not
explicit, infer carefully and mark the confidence or note the uncertainty.

The controlled extraction contract is defined by this profile.json content:
{profile_definition_json}

Extract every enabled field from profile.json. Populate the canonical profile
fields named by each field's maps_to value whenever possible. Also add one item
to controlled_metrics for every enabled profile.json field, using the same key
and label. For controlled_metrics.value, use a concise string for simple values
and compact JSON text for lists or objects.

For required fields with missing evidence, still include the controlled_metrics
item with an empty or unknown value, low confidence, and a clear uncertainty.
Separate non-negotiable requirements into hard_constraints and flexible ranking
preferences into soft_preferences.

Candidate input:
{cv_text}
"""


JOB_EXTRACTION_PROMPT = """\
You are CareerPilot, a job-search agent for international graduates in Germany.

Extract structured job objects from the pasted job descriptions. The input jobs
are separated by the literal delimiter ---JOB---. Preserve job order by assigning
job_index values starting at 1. Focus on target role, required/preferred skills,
minimum experience, seniority, location, remote policy, language requirements,
visa sponsorship or Blue Card signals, and any risks for an international
graduate.

Job descriptions:
{job_text}
"""


def ranking_prompt(
    profile_json: str,
    jobs_json: str,
    rubric: RankingRubric,
    top_n: int,
) -> str:
    """Build the job-ranking prompt."""

    return f"""\
You are CareerPilot, ranking jobs for an international graduate in Germany.

Use the rubric below to score each job from 0 to 100. Penalize risks that can
block a practical application, especially language requirements, seniority
mismatch, unclear visa friendliness, non-Germany location constraints, or missing
core skills. Be evidence-based: cite short evidence from the CV and job text.

Return the top {top_n} jobs only, sorted from strongest to weakest match.
For every job, include:
- matched_evidence from the CV and job description
- missing_skills
- risks
- uncertainties when evidence is unclear
- realistic_assessment such as Realistic, Possible but risky, or Risky
- next_action that the user can take immediately

Rubric:
{rubric.model_dump_json(indent=2)}

Candidate profile:
{profile_json}

Structured jobs:
{jobs_json}
"""


def evaluation_prompt(
    profile_json: str,
    jobs_json: str,
    ranking_json: str,
) -> str:
    """Build the evaluator prompt."""

    return f"""\
You are an evaluator for CareerPilot, a self-improving job-search agent for
international graduates in Germany.

Evaluate whether the ranking is useful and honest. Score each category from 1 to
5:
- fit_quality: whether the best jobs are truly aligned with the profile.
- risk_detection: whether visa, Blue Card, language, location, and seniority risks
  are identified clearly.
- actionability: whether the next actions are concrete and realistic.

Explain weaknesses directly. If a weak score is caused by missing evidence, say
what the ranking should inspect more carefully.

Candidate profile:
{profile_json}

Structured jobs:
{jobs_json}

Ranking:
{ranking_json}
"""


def rubric_improvement_prompt(
    current_rubric: RankingRubric,
    evaluation_json: str,
    profile_json: str,
    jobs: list[JobPosting],
) -> str:
    """Build the rubric-improvement prompt."""

    return f"""\
You are improving CareerPilot's job-ranking rubric.

The evaluator found weaknesses in the previous ranking. Create an improved
rubric that will produce a better rerun. Keep weights practical for international
graduates in Germany. Make the rubric stricter where risks were missed and more
specific where actions were vague.

Current rubric:
{current_rubric.model_dump_json(indent=2)}

Evaluation:
{evaluation_json}

Candidate profile:
{profile_json}

Jobs:
{[job.model_dump() for job in jobs]}
"""
