"""Candidate profile review step for CareerPilot."""

from __future__ import annotations

import streamlit as st

import app as careerpilot
from agent import CandidateProfile


careerpilot.setup_page("Candidate Profile - CareerPilot")

st.title("Candidate profile")

profile: CandidateProfile | None = st.session_state.get("profile")

if profile is None:
    st.info("Generate a profile from the CV upload page first.")
    careerpilot.render_page_button("Go to CV upload", "pages/1_CV_Upload.py", "profile_to_cv")
else:
    careerpilot.render_profile_summary(profile, expanded=True)

    cols = st.columns(2)
    with cols[0]:
        careerpilot.render_page_button("Update CV input", "pages/1_CV_Upload.py", "profile_update_cv")
    with cols[1]:
        careerpilot.render_page_button(
            "Continue to job ranking",
            "pages/3_Job_Ranking.py",
            "profile_to_ranking",
        )
