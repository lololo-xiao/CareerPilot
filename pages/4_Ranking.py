"""Job ranking step for CareerPilot."""

from __future__ import annotations

import streamlit as st

import app as careerpilot
from agent import CandidateProfile


model_name, top_n = careerpilot.setup_page("Ranking - CareerPilot")

st.title("Ranking")

profile: CandidateProfile | None = st.session_state.get("profile")
sourced_jobs: list[dict] = st.session_state.get("sourced_jobs") or []

if profile is None:
    st.info("Generate a candidate profile before ranking jobs.")
    careerpilot.render_page_button("Go to CV upload", "pages/1_CV_Upload.py", "ranking_to_cv")
elif not sourced_jobs:
    st.info("Source job links before ranking.")
    careerpilot.render_page_button("Go to sourcing", "pages/3_Sourcing.py", "ranking_to_sourcing")
else:
    st.caption("Select sourced jobs to rank against the saved candidate profile.")
    cols = st.columns(2)
    with cols[0]:
        careerpilot.render_page_button(
            "Review candidate profile",
            "pages/2_Candidate_Profile.py",
            "ranking_to_profile",
        )
    with cols[1]:
        careerpilot.render_page_button(
            "Update sourced jobs",
            "pages/3_Sourcing.py",
            "ranking_to_sourcing",
        )

    default_ids = [
        job["id"]
        for job in sourced_jobs
        if job.get("rank", True) and job.get("status") != "Archived"
    ]
    selected_ids = st.multiselect(
        "Jobs to rank",
        options=[job["id"] for job in sourced_jobs],
        default=default_ids,
        format_func=lambda job_id: careerpilot.sourced_job_label(
            next(job for job in sourced_jobs if job["id"] == job_id)
        ),
    )
    selected_jobs = careerpilot.selected_sourced_jobs(selected_ids)

    extra_job_details = st.text_area(
        "Additional job details",
        height=320,
        placeholder=(
            "Optional: paste full job descriptions for selected links. "
            "Separate multiple jobs with ---JOB---."
        ),
    )
    has_extracted_descriptions = any(job.get("description") for job in selected_jobs)
    if selected_jobs and not has_extracted_descriptions and not extra_job_details.strip():
        st.warning(
            "Ranking from links alone is limited. Extract job details on Sourcing or paste details here."
        )

    if st.button("Run job ranking", type="primary"):
        job_text = careerpilot.build_job_text_from_sourced_jobs(
            selected_jobs,
            extra_job_details,
        )
        if not selected_jobs:
            st.warning("Select at least one sourced job.")
        elif not job_text.strip():
            st.warning("Add job details or selected sourced jobs.")
        else:
            careerpilot.run_initial_ranking(profile, job_text, model_name, top_n)
            st.rerun()

    if st.session_state.get("initial_ranking") is not None:
        careerpilot.render_initial_section(top_n, include_profile=False)
        careerpilot.render_page_button(
            "Apply feedback and rerank",
            "pages/5_Feedback.py",
            "ranking_to_feedback",
        )
