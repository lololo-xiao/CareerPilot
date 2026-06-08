"""CV upload step for CareerPilot."""

from __future__ import annotations

import streamlit as st

import app as careerpilot


model_name, _ = careerpilot.setup_page("CV Upload - CareerPilot")

st.title("CV upload")

candidate_text = careerpilot.render_candidate_input_section()
profile_definition_json, profile_definition_error = careerpilot._load_profile_definition_json()
profile_is_current = careerpilot._profile_is_current(candidate_text)

if profile_definition_error:
    st.error(profile_definition_error)

profile_error = st.session_state.get("profile_error")
if profile_error:
    st.error(profile_error)

if candidate_text.strip():
    cols = st.columns([1, 3])
    with cols[0]:
        generate = st.button(
            "Generate profile",
            type="primary",
            disabled=profile_definition_error is not None,
        )
    with cols[1]:
        if profile_is_current:
            st.success("Profile is ready.")
            careerpilot.render_page_button(
                "Review candidate profile",
                "pages/2_Candidate_Profile.py",
                "cv_to_profile",
            )
        else:
            st.caption("Generate a candidate profile from this CV input.")

    if generate:
        if careerpilot.generate_candidate_profile(
            candidate_text,
            model_name,
            profile_definition_json or "{}",
        ):
            try:
                st.switch_page("pages/2_Candidate_Profile.py")
            except Exception:
                st.success("Profile generated. Use the sidebar to open Candidate Profile.")
else:
    st.info("Upload a CV or paste fallback CV text to begin.")
