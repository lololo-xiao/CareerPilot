"""End-to-end smoke test for the agentic loop (no Streamlit).

Runs the planner -> rank -> evaluate -> self-improve -> stop -> persist loop
against a tiny fixture, printing every streamed AgentStep.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv("/Users/wangxiao/Desktop/code/hackathon/.env")

from agent import CandidateProfile, JobPosting, LanguageProfile  # noqa: E402
from orchestrator import AgentStep, run_agent  # noqa: E402


def fixture() -> tuple[CandidateProfile, list[JobPosting]]:
    profile = CandidateProfile(
        target_roles=["Machine Learning Engineer", "NLP Engineer"],
        core_skills=["Python", "PyTorch", "LLMs", "NLP"],
        years_experience=1.0,
        education=["MSc Computer Science"],
        locations=["Berlin", "Remote Germany"],
        languages=[
            LanguageProfile(language="English", level="fluent"),
            LanguageProfile(language="German", level="A2"),
        ],
        visa_status="Needs visa sponsorship (non-EU)",
        seniority_level="junior",
        seniority_target="junior",
        hard_constraints=["Requires visa sponsorship", "German level only A2"],
        soft_preferences=["Berlin preferred", "LLM-focused role"],
    )
    jobs = [
        JobPosting(
            job_index=1,
            title="Junior ML Engineer",
            company="BerlinAI",
            location="Berlin",
            seniority="junior",
            required_skills=["Python", "PyTorch"],
            language_requirements=["English"],
            visa_blue_card_signals=["Visa sponsorship available"],
        ),
        JobPosting(
            job_index=2,
            title="Senior NLP Lead",
            company="DeutschCorp",
            location="Munich",
            seniority="senior",
            required_skills=["Python", "NLP", "Leadership"],
            minimum_years_experience=8,
            language_requirements=["German C1"],
            risks_for_international_graduate=["German C1 required", "Senior role"],
        ),
        JobPosting(
            job_index=3,
            title="Data Scientist (Working Student)",
            company="StartupX",
            location="Remote Germany",
            seniority="working student",
            required_skills=["Python", "SQL"],
            language_requirements=["English"],
        ),
    ]
    return profile, jobs


def main() -> None:
    profile, jobs = fixture()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def show(step: AgentStep) -> None:
        mark = {"ok": "[ok]", "warn": "[!]", "error": "[X]"}.get(step.status, "[ ]")
        print(f"{mark} {step.kind:9} | {step.title}")
        if step.detail:
            print(f"            {step.detail}")

    run = run_agent(profile, jobs, model_name=model, top_n=3, on_step=show)

    print("\n==== SUMMARY ====")
    print("plan        :", run.plan[:160])
    print("iterations  :", len(run.iterations))
    print("improved?   :", run.improved)
    print("stop_reason :", run.stop_reason)
    print("memory_used :", run.memory_used, "| memory_written:", run.memory_written)
    print("top match   :",
          run.final_ranking.matches[0].title if run.final_ranking.matches else "none",
          "score", run.final_ranking.matches[0].match_score if run.final_ranking.matches else "-")


if __name__ == "__main__":
    main()
