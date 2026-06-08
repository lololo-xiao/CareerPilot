"""CareerPilot Streamlit app."""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent import (
    CandidateProfile,
    CareerPilotAgent,
    JobMatch,
    JobPosting,
    RankingResult,
    default_rubric,
)
from constraints import (
    HumanFeedback,
    RerankedMatch,
    RerankingResult,
    apply_policy_to_ranking,
    build_policy_from_feedback,
)
from evaluators import EvaluationReport, evaluate_ranking
from observability import (
    TraceEvent,
    new_run_id,
    phoenix_status,
    record_human_feedback,
    setup_observability,
    trace_event,
)


load_dotenv()

MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def main() -> None:
    st.set_page_config(page_title="CareerPilot", page_icon=":briefcase:", layout="wide")
    setup_observability()
    _init_state()

    st.title("CareerPilot")
    st.caption(
        "Observable feedback loop: initial ranking -> human feedback -> constraints -> reranking"
    )

    with st.sidebar:
        st.header("Demo controls")
        model_choice = st.selectbox(
            "Gemini model",
            options=MODEL_OPTIONS + ["custom"],
            index=MODEL_OPTIONS.index(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
            if os.getenv("GEMINI_MODEL", "gemini-2.5-flash") in MODEL_OPTIONS
            else len(MODEL_OPTIONS),
        )
        model_name = (
            st.text_input("Custom model", value=os.getenv("GEMINI_MODEL", ""))
            if model_choice == "custom"
            else model_choice
        )
        top_n = st.slider("Jobs to show", min_value=1, max_value=10, value=5)
        st.caption(phoenix_status())

        if st.button("Reset demo"):
            _reset_state()
            st.rerun()

    st.subheader("Candidate input")
    uploaded_cv = st.file_uploader("CV PDF", type=["pdf"])
    pdf_text, pdf_error = _extract_uploaded_pdf_text(uploaded_cv)

    if uploaded_cv is not None:
        if pdf_text:
            word_count = len(pdf_text.split())
            st.success(f"Extracted {word_count} words from {uploaded_cv.name}.")
            with st.expander("Extracted CV text preview", expanded=False):
                st.text_area("Extracted PDF text", value=pdf_text, height=180, disabled=True)
        elif pdf_error:
            st.warning(pdf_error)

    pasted_cv_text = st.text_area(
        "Fallback pasted CV text",
        height=180,
        placeholder="Paste CV text if PDF extraction is unavailable or looks incomplete",
    )
    prefer_pasted_cv = False
    if pdf_text and pasted_cv_text.strip():
        prefer_pasted_cv = st.checkbox("Use pasted CV fallback instead of extracted PDF text")

    extra_notes = st.text_area(
        "Extra candidate notes",
        height=140,
        placeholder=(
            "Preferences, constraints, target roles, visa needs, location preferences, "
            "personal context, or anything missing from the CV"
        ),
    )
    candidate_text = _build_candidate_input(
        pdf_text=pdf_text,
        pasted_cv_text=pasted_cv_text,
        extra_notes=extra_notes,
        prefer_pasted_cv=prefer_pasted_cv,
    )

    job_text = st.text_area(
        "Job descriptions",
        height=280,
        placeholder="Paste jobs separated by ---JOB---",
    )

    if st.button("1. Run initial ranking", type="primary"):
        if not candidate_text.strip() or not job_text.strip():
            st.warning("Add candidate input and job descriptions.")
        else:
            run_initial_ranking(candidate_text, job_text, model_name, top_n)

    if st.session_state.get("initial_ranking") is not None:
        render_initial_section(top_n)
        render_feedback_section()

    if st.session_state.get("reranking") is not None:
        render_comparison_section()

    render_trace_section()


def run_initial_ranking(
    candidate_text: str,
    job_text: str,
    model_name: str,
    top_n: int,
) -> None:
    run_id = new_run_id()
    events: list[TraceEvent] = []
    st.session_state["run_id"] = run_id
    st.session_state["trace_events"] = events
    st.session_state["reranking"] = None
    st.session_state["feedback"] = None
    st.session_state["policy"] = None

    agent = CareerPilotAgent(model_name=model_name)
    rubric = default_rubric()

    try:
        with st.status("Running initial agent pass", expanded=True) as status:
            st.write("Extracting profile")
            profile = agent.extract_profile(candidate_text)
            trace_event(
                events,
                run_id,
                "profile_extraction",
                "Structured profile extracted from candidate input.",
                {"target_roles": profile.target_roles, "visa_status": profile.visa_status},
            )

            st.write("Extracting jobs")
            jobs = agent.extract_jobs(job_text)
            trace_event(
                events,
                run_id,
                "job_extraction",
                f"Extracted {len(jobs)} job objects.",
                {"job_count": len(jobs)},
            )
            if not jobs:
                st.error("No jobs were extracted from the pasted descriptions.")
                return

            st.write("Ranking jobs")
            ranking = agent.rank_jobs(profile, jobs, rubric, top_n=min(len(jobs), 20))
            trace_event(
                events,
                run_id,
                "initial_ranking",
                "Initial Gemini ranking completed.",
                {"matches": len(ranking.matches)},
            )

            evaluation = _safe_evaluate(agent, profile, jobs, ranking, events, run_id)
            status.update(label="Initial ranking ready", state="complete")

        st.session_state["profile"] = profile
        st.session_state["jobs"] = jobs
        st.session_state["initial_ranking"] = ranking
        st.session_state["evaluation"] = evaluation
        st.session_state["top_n"] = top_n
    except Exception as exc:
        trace_event(events, run_id, "initial_run_error", str(exc), status="error")
        st.error(f"CareerPilot failed: {exc}")
    finally:
        agent.close()


def _extract_uploaded_pdf_text(uploaded_cv: Any | None) -> tuple[str, str | None]:
    if uploaded_cv is None:
        return "", None

    try:
        return _extract_pdf_text(uploaded_cv.getvalue()), None
    except Exception as exc:
        return "", f"PDF text extraction failed: {exc}"


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to parse uploaded PDFs.") from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        decrypt_result = reader.decrypt("")
        if decrypt_result == 0:
            raise ValueError("Encrypted PDFs are not supported yet. Use pasted CV text.")

    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"[Page {page_number}]\n{page_text}")

    extracted_text = "\n\n".join(pages).strip()
    if not extracted_text:
        raise ValueError(
            "No selectable text was found in the PDF. Use pasted CV text while OCR support is added."
        )
    return extracted_text


def _build_candidate_input(
    *,
    pdf_text: str,
    pasted_cv_text: str,
    extra_notes: str,
    prefer_pasted_cv: bool,
) -> str:
    sections: list[tuple[str, str]] = []

    cv_source = pasted_cv_text if prefer_pasted_cv or not pdf_text.strip() else pdf_text
    cv_source_label = (
        "Fallback pasted CV text"
        if prefer_pasted_cv or not pdf_text.strip()
        else "CV text extracted from uploaded PDF"
    )
    if cv_source.strip():
        sections.append((cv_source_label, cv_source.strip()))

    if extra_notes.strip():
        sections.append(("Additional candidate notes from the user", extra_notes.strip()))

    return "\n\n".join(f"{label}:\n{text}" for label, text in sections)


def _safe_evaluate(
    agent: CareerPilotAgent,
    profile: CandidateProfile,
    jobs: list[JobPosting],
    ranking: RankingResult,
    events: list[TraceEvent],
    run_id: str,
) -> EvaluationReport | None:
    try:
        evaluation = evaluate_ranking(agent, profile, jobs, ranking)
        trace_event(
            events,
            run_id,
            "evaluator_result",
            "Evaluator scored the initial ranking.",
            evaluation.model_dump(),
        )
        return evaluation
    except Exception as exc:
        trace_event(
            events,
            run_id,
            "evaluator_result",
            f"Evaluator skipped: {exc}",
            status="warning",
        )
        return None


def render_initial_section(top_n: int) -> None:
    ranking: RankingResult = st.session_state["initial_ranking"]
    evaluation: EvaluationReport | None = st.session_state.get("evaluation")

    st.header("Initial ranking")
    st.caption("This is Gemini's first pass before your feedback constraints are applied.")
    render_ranking_cards(ranking.matches[:top_n])

    if evaluation is not None:
        with st.expander("Evaluator notes", expanded=False):
            cols = st.columns(3)
            cols[0].metric("Fit quality", evaluation.fit_quality)
            cols[1].metric("Risk detection", evaluation.risk_detection)
            cols[2].metric("Actionability", evaluation.actionability)
            st.write(evaluation.explanation)
            if evaluation.weaknesses:
                st.write(evaluation.weaknesses)


def render_feedback_section() -> None:
    profile: CandidateProfile = st.session_state["profile"]
    default_years = int((profile.years_experience or 1) + 1)

    st.header("Human feedback")
    st.caption("Tell CareerPilot what is truly acceptable. These answers become explicit reranking constraints.")

    with st.form("feedback_form"):
        free_text = st.text_area(
            "Global feedback",
            placeholder="Example: Visa sponsorship is a hard requirement. I do not accept German C1 jobs. I prefer junior or mid-level AI engineer roles.",
            height=110,
        )

        col1, col2 = st.columns(2)
        with col1:
            visa_required = st.checkbox("Visa sponsorship / Blue Card friendliness is mandatory")
            accepted_german_level = st.selectbox(
                "Highest German requirement I accept",
                ["", "A1", "A2", "B1", "B2", "C1", "C2"],
                index=2,
            )
            reject_german = st.checkbox("Strongly cap jobs above this German level", value=True)
        with col2:
            accepted_locations = st.text_input("Acceptable locations", value="Berlin, Munich")
            preferred_seniority = st.multiselect(
                "Preferred seniority",
                ["Internship", "Working student", "Junior", "Mid-level", "Senior", "Lead"],
                default=["Junior", "Mid-level"],
            )
            max_years = st.number_input(
                "Maximum realistic required years of experience",
                min_value=0.0,
                max_value=20.0,
                value=float(default_years),
                step=0.5,
            )

        prioritize_realistic = st.checkbox(
            "Prioritize realistic jobs over prestigious titles",
            value=True,
        )
        submitted = st.form_submit_button("2. Update constraints and rerank", type="primary")

    if submitted:
        feedback = HumanFeedback(
            free_text=free_text,
            visa_sponsorship_mandatory=visa_required,
            accepted_german_level=accepted_german_level,
            reject_german_above_level=reject_german,
            acceptable_locations=_split_csv(accepted_locations),
            preferred_seniority=preferred_seniority,
            max_realistic_years=max_years,
            prioritize_realistic_jobs=prioritize_realistic,
        )
        rerank_with_feedback(feedback)
        st.rerun()


def rerank_with_feedback(feedback: HumanFeedback) -> None:
    profile: CandidateProfile = st.session_state["profile"]
    jobs: list[JobPosting] = st.session_state["jobs"]
    ranking: RankingResult = st.session_state["initial_ranking"]
    events: list[TraceEvent] = st.session_state["trace_events"]
    run_id: str = st.session_state["run_id"]

    policy = build_policy_from_feedback(feedback)
    record_human_feedback(events, run_id, feedback.model_dump(), policy.model_dump())
    trace_event(
        events,
        run_id,
        "constraint_update",
        "Human feedback converted into explicit hard rules and preferences.",
        policy.model_dump(),
    )
    reranking = apply_policy_to_ranking(profile, jobs, ranking, policy)
    trace_event(
        events,
        run_id,
        "improved_reranking",
        "Ranking rerun with explicit user constraints and score caps.",
        {"matches": [match.model_dump() for match in reranking.matches]},
    )

    st.session_state["feedback"] = feedback
    st.session_state["policy"] = policy
    st.session_state["reranking"] = reranking


def render_comparison_section() -> None:
    reranking: RerankingResult = st.session_state["reranking"]

    st.header("Before / after reranking")
    st.caption("The second pass uses your feedback as explicit constraints, including hard score caps.")

    with st.expander("Updated ranking policy", expanded=True):
        if reranking.policy.hard_rules:
            st.write("Hard rules")
            for rule in reranking.policy.hard_rules:
                st.markdown(f"- {rule}")
        if reranking.policy.soft_preferences:
            st.write("Soft preferences")
            for preference in reranking.policy.soft_preferences:
                st.markdown(f"- {preference}")

    for match in reranking.matches[: st.session_state.get("top_n", 5)]:
        render_comparison_card(match)


def render_ranking_cards(matches: list[JobMatch]) -> None:
    if not matches:
        st.info("No ranked jobs returned.")
        return

    for match in matches:
        with st.container(border=True):
            cols = st.columns([1, 3, 1])
            cols[0].metric("Score", match.match_score)
            title = match.title if not match.company else f"{match.title} at {match.company}"
            cols[1].subheader(title)
            cols[1].caption(match.location or "Location unclear")
            assessment = getattr(match, "realistic_assessment", "")
            cols[2].write(assessment or _assessment_from_score(match.match_score))

            _render_list("Why it matches", match.matched_evidence)
            _render_list("Risks", match.risks)
            _render_list("Uncertainties", getattr(match, "uncertainties", []))
            st.write("Next action")
            st.write(match.next_action)


def render_comparison_card(match: RerankedMatch) -> None:
    with st.container(border=True):
        cols = st.columns([1, 1, 1, 2])
        cols[0].metric("Initial", match.initial_score)
        cols[1].metric("After feedback", match.improved_score, delta=match.score_delta)
        cols[2].write(match.final_recommendation)
        title = match.title if not match.company else f"{match.title} at {match.company}"
        cols[3].subheader(title)
        cols[3].caption(match.location or "Location unclear")

        st.write(match.change_reason)
        _render_list("Applied constraints", match.applied_constraints or ["No new constraint changed this job."])
        _render_list("Risks still visible", match.risks)
        _render_list("Uncertainties", match.uncertainties)
        st.write("Next action")
        st.write(match.next_action)


def render_trace_section() -> None:
    events: list[TraceEvent] = st.session_state.get("trace_events", [])
    if not events:
        return

    st.header("Traced steps")
    st.caption("Local trace for the demo. Phoenix/Arize export can be attached through observability.py.")
    for event in events:
        with st.expander(f"{event.step} - {event.status}", expanded=False):
            st.write(event.message)
            st.caption(event.timestamp)
            if event.metadata:
                st.json(event.metadata)


def _render_list(label: str, values: list[str]) -> None:
    st.write(label)
    if values:
        for value in values:
            st.markdown(f"- {value}")
    else:
        st.caption("None reported")


def _assessment_from_score(score: int) -> str:
    if score >= 75:
        return "Realistic"
    if score >= 60:
        return "Possible but risky"
    return "Risky"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "run_id": None,
        "trace_events": [],
        "profile": None,
        "jobs": None,
        "initial_ranking": None,
        "evaluation": None,
        "feedback": None,
        "policy": None,
        "reranking": None,
        "top_n": 5,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_state() -> None:
    for key in [
        "run_id",
        "trace_events",
        "profile",
        "jobs",
        "initial_ranking",
        "evaluation",
        "feedback",
        "policy",
        "reranking",
    ]:
        st.session_state[key] = [] if key == "trace_events" else None


if __name__ == "__main__":
    main()
