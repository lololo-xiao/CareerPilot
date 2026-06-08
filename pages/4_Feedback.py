"""Feedback and reranking step for CareerPilot."""

from __future__ import annotations

import streamlit as st

import app as careerpilot


_, top_n = careerpilot.setup_page("Feedback - CareerPilot")

st.title("Feedback and reranking")

if st.session_state.get("initial_ranking") is None:
    st.info("Run job ranking before applying feedback.")
    careerpilot.render_page_button("Go to job ranking", "pages/3_Job_Ranking.py", "feedback_to_ranking")
else:
    careerpilot.render_feedback_section()

    if st.session_state.get("reranking") is not None:
        careerpilot.render_comparison_section()
    else:
        careerpilot.render_page_button("Back to job ranking", "pages/3_Job_Ranking.py", "feedback_back_ranking")
