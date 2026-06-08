"""Job sourcing step for CareerPilot."""

from __future__ import annotations

import streamlit as st

import app as careerpilot


careerpilot.setup_page("Sourcing - CareerPilot")

st.title("Sourcing")
st.caption("Build the pool of job links that can later be selected for ranking.")

uploaded_links = st.file_uploader("Upload job links", type=["txt", "csv"])
uploaded_text = ""
if uploaded_links is not None:
    uploaded_text = uploaded_links.getvalue().decode("utf-8", errors="ignore")
    urls = careerpilot.parse_job_urls(uploaded_text)
    st.caption(f"Found {len(urls)} links in {uploaded_links.name}.")

pasted_links = st.text_area(
    "Paste job links",
    height=180,
    placeholder="One job link per line, or paste copied search results containing URLs.",
)

source_label = st.text_input("Source label", value="Manual sourcing")

if st.button("Add links to sourcing pool", type="primary"):
    urls = careerpilot.parse_job_urls("\n".join([uploaded_text, pasted_links]))
    if not urls:
        st.warning("Add at least one valid http or https job link.")
    else:
        added, duplicates = careerpilot.add_sourced_jobs(urls, source_label.strip() or "Manual sourcing")
        st.success(f"Added {added} jobs. Skipped {duplicates} duplicates.")
        st.rerun()

st.subheader("Sourced job pool")
careerpilot.render_sourced_jobs_editor()

pool = st.session_state.get("sourced_jobs") or []
if pool:
    careerpilot.render_page_button(
        "Continue to ranking",
        "pages/4_Ranking.py",
        "sourcing_to_ranking",
    )
