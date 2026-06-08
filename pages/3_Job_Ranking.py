"""Job ranking step for CareerPilot."""

from __future__ import annotations

import streamlit as st

import app as careerpilot
from agent import CandidateProfile


model_name, top_n = careerpilot.setup_page("Job Ranking - CareerPilot")

st.title("Job ranking")

profile: CandidateProfile | None = st.session_state.get("profile")

if profile is None:
    st.info("Generate a candidate profile before ranking jobs.")
    careerpilot.render_page_button("Go to CV upload", "pages/1_CV_Upload.py", "ranking_to_cv")
else:
    st.caption("Ranking uses the saved candidate profile.")
    careerpilot.render_page_button(
        "Review candidate profile",
        "pages/2_Candidate_Profile.py",
        "ranking_to_profile",
    )

    job_text = st.text_area(
        "Job descriptions",
        height=320,
        placeholder="Paste jobs separated by ---JOB---",
    )

    if st.button("Run job ranking", type="primary"):
        if not job_text.strip():
            st.warning("Add job descriptions.")
        else:
            careerpilot.run_initial_ranking(profile, job_text, model_name, top_n)
            st.rerun()

    if st.session_state.get("initial_ranking") is not None:
        careerpilot.render_initial_section(top_n, include_profile=False)
        careerpilot.render_page_button(
            "Apply feedback and rerank",
            "pages/4_Feedback.py",
            "ranking_to_feedback",
        )
