"""Human feedback constraints and deterministic reranking."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agent import CandidateProfile, JobMatch, JobPosting, RankingResult


GERMAN_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


class ConstraintPolicy(BaseModel):
    visa_sponsorship_mandatory: bool = False
    accepted_german_level: str = ""
    reject_german_above_level: bool = False
    acceptable_locations: list[str] = Field(default_factory=list)
    preferred_seniority: list[str] = Field(default_factory=list)
    max_realistic_years: float | None = None
    prioritize_realistic_jobs: bool = True
    hard_rules: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)


class HumanFeedback(BaseModel):
    free_text: str = ""
    visa_sponsorship_mandatory: bool = False
    accepted_german_level: str = ""
    reject_german_above_level: bool = False
    acceptable_locations: list[str] = Field(default_factory=list)
    preferred_seniority: list[str] = Field(default_factory=list)
    max_realistic_years: float | None = None
    prioritize_realistic_jobs: bool = True


class RerankedMatch(BaseModel):
    job_index: int
    title: str
    company: str = ""
    location: str = ""
    initial_score: int
    improved_score: int
    score_delta: int
    matched_evidence: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    next_action: str = ""
    applied_constraints: list[str] = Field(default_factory=list)
    score_cap: int | None = None
    change_reason: str = ""
    final_recommendation: str = ""
    realistic_assessment: str = ""


class RerankingResult(BaseModel):
    policy: ConstraintPolicy
    matches: list[RerankedMatch] = Field(default_factory=list)


def build_policy_from_feedback(feedback: HumanFeedback) -> ConstraintPolicy:
    """Convert user feedback into explicit ranking constraints."""

    text = feedback.free_text.lower()
    policy = ConstraintPolicy(
        visa_sponsorship_mandatory=feedback.visa_sponsorship_mandatory,
        accepted_german_level=feedback.accepted_german_level,
        reject_german_above_level=feedback.reject_german_above_level,
        acceptable_locations=feedback.acceptable_locations,
        preferred_seniority=feedback.preferred_seniority,
        max_realistic_years=feedback.max_realistic_years,
        prioritize_realistic_jobs=feedback.prioritize_realistic_jobs,
    )

    if re.search(r"\bvisa\b.*\b(hard|required|mandatory|must|need)", text):
        policy.visa_sponsorship_mandatory = True
    if "blue card" in text and re.search(r"\b(hard|required|mandatory|must|need)", text):
        policy.visa_sponsorship_mandatory = True

    german_match = re.search(r"german\s*(a1|a2|b1|b2|c1|c2)", text)
    if german_match and (
        "do not accept" in text
        or "don't accept" in text
        or "not accept" in text
        or "hard reject" in text
    ):
        policy.reject_german_above_level = True

    for city in ["berlin", "munich", "hamburg", "cologne", "frankfurt", "stuttgart"]:
        if re.search(rf"\b(can accept|accept|ok|okay|willing).{{0,30}}\b{city}\b", text):
            normalized = city.title()
            if normalized not in policy.acceptable_locations:
                policy.acceptable_locations.append(normalized)

    if "junior" in text and "Junior" not in policy.preferred_seniority:
        policy.preferred_seniority.append("Junior")
    if re.search(r"\bmid\b|mid-level", text) and "Mid-level" not in policy.preferred_seniority:
        policy.preferred_seniority.append("Mid-level")
    if "realistic" in text:
        policy.prioritize_realistic_jobs = True

    policy.hard_rules = _hard_rules(policy)
    policy.soft_preferences = _soft_preferences(policy)
    return policy


def apply_policy_to_ranking(
    profile: CandidateProfile,
    jobs: list[JobPosting],
    ranking: RankingResult,
    policy: ConstraintPolicy,
) -> RerankingResult:
    """Apply explicit constraints and score caps to an initial ranking."""

    jobs_by_index = {job.job_index: job for job in jobs}
    reranked = [
        _apply_policy_to_match(profile, jobs_by_index.get(match.job_index), match, policy)
        for match in ranking.matches
    ]
    reranked.sort(key=lambda item: item.improved_score, reverse=True)
    return RerankingResult(policy=policy, matches=reranked)


def _apply_policy_to_match(
    profile: CandidateProfile,
    job: JobPosting | None,
    match: JobMatch,
    policy: ConstraintPolicy,
) -> RerankedMatch:
    score = match.match_score
    cap: int | None = None
    applied: list[str] = []

    if job is not None and policy.visa_sponsorship_mandatory and not _is_visa_friendly(job):
        cap = _lower_cap(cap, 50)
        applied.append("Visa sponsorship is mandatory, but this job has no clear visa support.")

    if job is not None and policy.reject_german_above_level:
        required_level = _required_german_level(job)
        if required_level and _german_level_too_high(required_level, policy.accepted_german_level):
            cap = _lower_cap(cap, 55)
            applied.append(
                f"German {required_level} appears required, above accepted level {policy.accepted_german_level or 'unspecified'}."
            )

    if job is not None and policy.max_realistic_years is not None:
        if _requires_too_much_experience(job, policy.max_realistic_years):
            cap = _lower_cap(cap, 60)
            applied.append(
                f"Experience requirement exceeds the realistic cap of {policy.max_realistic_years:g} years."
            )

    if job is not None and policy.acceptable_locations and not _location_is_acceptable(job, policy):
        cap = _lower_cap(cap, 65)
        applied.append("Location is not in the accepted list and no clear remote option is present.")

    if job is not None and policy.preferred_seniority:
        if _seniority_is_not_preferred(job, policy.preferred_seniority):
            cap = _lower_cap(cap, 70)
            applied.append("Seniority does not match the preferred junior/mid-level focus.")

    improved_score = min(score, cap) if cap is not None else score
    if policy.prioritize_realistic_jobs and _has_major_risk(match):
        improved_score = max(0, improved_score - 5)
        applied.append("Realistic jobs are prioritized, so unresolved major risks reduce the score.")

    delta = improved_score - score
    return RerankedMatch(
        job_index=match.job_index,
        title=match.title,
        company=match.company,
        location=match.location,
        initial_score=score,
        improved_score=improved_score,
        score_delta=delta,
        matched_evidence=match.matched_evidence,
        missing_skills=match.missing_skills,
        risks=match.risks,
        uncertainties=getattr(match, "uncertainties", []),
        next_action=match.next_action,
        applied_constraints=applied,
        score_cap=cap,
        change_reason=_change_reason(delta, applied),
        final_recommendation=_recommendation(improved_score, applied),
        realistic_assessment=_realistic_assessment(improved_score, applied),
    )


def _hard_rules(policy: ConstraintPolicy) -> list[str]:
    rules: list[str] = []
    if policy.visa_sponsorship_mandatory:
        rules.append("Visa sponsorship / Blue Card friendliness is mandatory.")
    if policy.reject_german_above_level:
        rules.append(
            f"Reject or strongly cap jobs requiring German above {policy.accepted_german_level or 'the accepted level'}."
        )
    if policy.max_realistic_years is not None:
        rules.append(f"Cap roles requiring more than {policy.max_realistic_years:g} years.")
    if policy.acceptable_locations:
        rules.append(f"Accept only these locations unless remote is clear: {', '.join(policy.acceptable_locations)}.")
    return rules


def _soft_preferences(policy: ConstraintPolicy) -> list[str]:
    preferences: list[str] = []
    if policy.preferred_seniority:
        preferences.append(f"Prefer seniority: {', '.join(policy.preferred_seniority)}.")
    if policy.prioritize_realistic_jobs:
        preferences.append("Prioritize realistic applications over prestigious titles.")
    return preferences


def _lower_cap(existing: int | None, candidate: int) -> int:
    if existing is None:
        return candidate
    return min(existing, candidate)


def _is_visa_friendly(job: JobPosting) -> bool:
    text = _job_text(job)
    negative = ["no visa", "no sponsorship", "cannot sponsor", "not sponsor", "must already have work authorization"]
    positive = ["visa", "sponsorship", "blue card", "relocation", "work permit"]
    if any(item in text for item in negative):
        return False
    return any(item in text for item in positive)


def _required_german_level(job: JobPosting) -> str:
    text = " ".join(job.language_requirements).upper()
    for level in reversed(GERMAN_LEVELS):
        if level in text and "GERMAN" in text:
            return level
    if "GERMAN" in text and ("FLUENT" in text or "NATIVE" in text):
        return "C1"
    return ""


def _german_level_too_high(required: str, accepted: str) -> bool:
    if not accepted:
        return True
    try:
        return GERMAN_LEVELS.index(required) > GERMAN_LEVELS.index(accepted)
    except ValueError:
        return True


def _requires_too_much_experience(job: JobPosting, max_years: float) -> bool:
    if job.minimum_years_experience is not None and job.minimum_years_experience > max_years:
        return True
    text = _job_text(job)
    return any(term in text for term in ["senior", "lead", "principal", "staff engineer"])


def _location_is_acceptable(job: JobPosting, policy: ConstraintPolicy) -> bool:
    location = job.location.lower()
    remote = job.remote_policy.lower()
    if "remote" in remote or "remote" in location:
        return True
    return any(city.lower() in location for city in policy.acceptable_locations)


def _seniority_is_not_preferred(job: JobPosting, preferred: list[str]) -> bool:
    seniority = f"{job.seniority} {job.title}".lower()
    preferred_text = " ".join(preferred).lower()
    if any(term in seniority for term in ["senior", "lead", "principal", "staff"]):
        return not ("senior" in preferred_text or "lead" in preferred_text)
    if "junior" in seniority:
        return "junior" not in preferred_text
    if "mid" in seniority:
        return "mid" not in preferred_text
    return False


def _has_major_risk(match: JobMatch) -> bool:
    uncertainties = getattr(match, "uncertainties", [])
    text = " ".join(match.risks + uncertainties).lower()
    return any(term in text for term in ["visa", "german", "senior", "experience", "location", "unclear"])


def _change_reason(delta: int, applied: list[str]) -> str:
    if not applied:
        return "No new hard constraints changed this score."
    if delta < 0:
        return "Score decreased because user feedback added stricter constraints."
    return "Constraints were checked; score stayed stable."


def _recommendation(score: int, applied: list[str]) -> str:
    critical = any("mandatory" in item.lower() or "above accepted" in item.lower() for item in applied)
    if score >= 75 and not critical:
        return "Apply now"
    if score >= 60:
        return "Apply later"
    return "Skip"


def _realistic_assessment(score: int, applied: list[str]) -> str:
    if score >= 75 and not applied:
        return "Realistic"
    if score >= 60:
        return "Possible but risky"
    return "Risky"


def _job_text(job: JobPosting) -> str:
    return " ".join(
        [
            job.title,
            job.company,
            job.location,
            job.remote_policy,
            job.seniority,
            " ".join(job.required_skills),
            " ".join(job.preferred_skills),
            " ".join(job.language_requirements),
            " ".join(job.visa_blue_card_signals),
            " ".join(job.risks_for_international_graduate),
            job.raw_summary,
        ]
    ).lower()
