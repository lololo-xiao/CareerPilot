"""CareerPilot Streamlit app."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

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
PROFILE_DEFINITION_PATH = Path(__file__).with_name("profile.json")
JOB_URL_RE = re.compile(r"https?://[^\s,;]+")
JOB_STATUSES = [
    "Sourced",
    "Interested",
    "Selected for ranking",
    "Applied",
    "Interviewing",
    "Rejected",
    "Archived",
]
SOURCE_JOB_COLUMN_ORDER = [
    "Rank",
    "Title",
    "Company",
    "Location",
    "Seniority",
    "Status",
    "Extraction",
    "URL",
    "Description",
    "Notes",
    "Source",
]


def setup_page(page_title: str = "CareerPilot") -> tuple[str, int]:
    st.set_page_config(page_title=page_title, page_icon=":briefcase:", layout="wide")
    setup_observability()
    _init_state()

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

    return model_name, top_n


def main() -> None:
    setup_page()

    st.title("CareerPilot")
    st.caption("Step-based job matching workflow for international graduates in Germany.")

    render_workflow_status()


def render_workflow_status() -> None:
    profile_ready = st.session_state.get("profile") is not None
    job_pool_count = len(st.session_state.get("job_pool") or st.session_state.get("sourced_jobs") or [])
    ranking_ready = st.session_state.get("initial_ranking") is not None
    reranking_ready = st.session_state.get("reranking") is not None

    cols = st.columns(5)
    cols[0].container(border=True).write("CV upload")
    cols[0].caption("Add CV text, PDF, and notes.")

    cols[1].container(border=True).write(
        "Candidate profile ready" if profile_ready else "Candidate profile pending"
    )
    cols[1].caption("Generate and review the structured profile.")

    cols[2].container(border=True).write(
        f"{job_pool_count} jobs in pool" if job_pool_count else "Job pool pending"
    )
    cols[2].caption("Build a pool of jobs.")

    cols[3].container(border=True).write(
        "Ranking ready" if ranking_ready else "Job ranking pending"
    )
    cols[3].caption("Rank parsed jobs from the pool.")

    cols[4].container(border=True).write(
        "Feedback applied" if reranking_ready else "Feedback pending"
    )
    cols[4].caption("Add constraints and rerank.")

    render_page_button("Start with CV upload", "pages/1_CV_Upload.py", "home_cv")
    render_page_button(
        "Review candidate profile",
        "pages/2_Candidate_Profile.py",
        "home_profile",
    )
    render_page_button("Build job pool", "pages/3_Sourcing.py", "home_sourcing")
    render_page_button("Rank jobs", "pages/4_Ranking.py", "home_ranking")
    render_page_button("Apply feedback", "pages/5_Feedback.py", "home_feedback")


def render_page_button(label: str, page: str, key: str) -> None:
    if st.button(label, key=key):
        try:
            st.switch_page(page)
        except Exception:
            st.warning("Use the sidebar page navigation for this step.")


def parse_job_urls(raw_text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in JOB_URL_RE.findall(raw_text):
        url = match.strip().rstrip(").,;]")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def add_sourced_jobs(urls: list[str], source: str) -> tuple[int, int]:
    pool: list[dict[str, Any]] = st.session_state.setdefault("sourced_jobs", [])
    existing_ids = {job["id"] for job in pool}
    added = 0
    duplicates = 0

    for url in urls:
        job_id = _job_source_id(url)
        if job_id in existing_ids:
            duplicates += 1
            continue
        pool.append(
            {
                "id": job_id,
                "url": url,
                "title": _title_from_url(url),
                "company": "",
                "location": "",
                "seniority": "",
                "description": "",
                "source": source,
                "status": "Sourced",
                "extraction_status": "Not extracted",
                "notes": "",
                "rank": True,
            }
        )
        existing_ids.add(job_id)
        added += 1

    st.session_state["sourced_jobs"] = pool
    return added, duplicates


def render_sourced_jobs_editor() -> None:
    pool: list[dict[str, Any]] = st.session_state.get("sourced_jobs") or []
    if not pool:
        st.info("No sourced jobs yet.")
        return

    rows = [
        {
            "Rank": bool(job.get("rank", True)),
            "Title": job.get("title", ""),
            "Company": job.get("company", ""),
            "Location": job.get("location", ""),
            "Seniority": job.get("seniority", ""),
            "URL": job.get("url", ""),
            "Status": job.get("status", "Sourced"),
            "Extraction": job.get("extraction_status", "Not extracted"),
            "Description": job.get("description", ""),
            "Notes": job.get("notes", ""),
            "id": job.get("id", ""),
            "Source": job.get("source", ""),
        }
        for job in pool
    ]

    edited_rows = st.data_editor(
        rows,
        hide_index=True,
        use_container_width=True,
        column_order=SOURCE_JOB_COLUMN_ORDER,
        column_config={
            "Rank": st.column_config.CheckboxColumn("Rank", help="Include by default on the Ranking page."),
            "URL": st.column_config.LinkColumn("URL"),
            "Status": st.column_config.SelectboxColumn("Status", options=JOB_STATUSES),
            "Extraction": st.column_config.TextColumn("Extraction"),
            "Description": st.column_config.TextColumn("Description"),
            "Notes": st.column_config.TextColumn("Notes"),
            "Source": st.column_config.TextColumn("Source"),
        },
        disabled=["Extraction", "Source"],
        key="sourced_jobs_editor",
    )

    if st.button("Save sourcing changes"):
        by_id = {job["id"]: job for job in pool}
        for row in edited_rows:
            job = by_id.get(row["id"])
            if not job:
                continue
            job["rank"] = bool(row["Rank"])
            job["title"] = row["Title"].strip() or _title_from_url(row["URL"])
            job["company"] = row["Company"].strip()
            job["location"] = row["Location"].strip()
            job["seniority"] = row["Seniority"].strip()
            job["url"] = row["URL"].strip()
            job["status"] = row["Status"]
            job["description"] = row["Description"].strip()
            job["notes"] = row["Notes"].strip()
        st.session_state["sourced_jobs"] = pool
        st.success("Sourcing pool updated.")


def enrich_sourced_jobs(job_ids: list[str], model_name: str) -> tuple[int, int]:
    pool: list[dict[str, Any]] = st.session_state.get("sourced_jobs") or []
    selected_ids = set(job_ids)
    jobs_to_enrich = [job for job in pool if job.get("id") in selected_ids]
    if not jobs_to_enrich:
        return 0, 0

    agent = CareerPilotAgent(model_name=model_name)
    updated = 0
    failed = 0
    try:
        with st.status("Extracting job details from links", expanded=True) as status:
            for job in jobs_to_enrich:
                url = job.get("url", "")
                st.write(f"Reading {url}")
                try:
                    page_title, page_text = fetch_job_page_text(url)
                    if page_title:
                        job["page_title"] = page_title

                    extracted_jobs = agent.extract_jobs(
                        "\n\n".join(
                            [
                                f"Job source URL: {url}",
                                f"Page title: {page_title}",
                                "Extract one job posting from the page text below.",
                                page_text,
                            ]
                        )
                    )
                    if not extracted_jobs:
                        raise RuntimeError("No job posting was extracted from the page.")

                    posting = extracted_jobs[0]
                    _apply_extracted_job_to_source(job, posting, page_title)
                    updated += 1
                except Exception as exc:
                    job["extraction_status"] = f"Needs manual details: {exc}"
                    failed += 1

            st.session_state["sourced_jobs"] = pool
            status.update(
                label=f"Extraction finished: {updated} updated, {failed} need manual details",
                state="complete" if failed == 0 else "error",
            )
    finally:
        agent.close()

    return updated, failed


def selected_sourced_jobs(selected_ids: list[str]) -> list[dict[str, Any]]:
    selected = set(selected_ids)
    return [
        job
        for job in st.session_state.get("sourced_jobs", [])
        if job.get("id") in selected
    ]


def sourced_job_label(job: dict[str, Any]) -> str:
    title = job.get("title") or _title_from_url(job.get("url", ""))
    return f"{title} - {urlparse(job.get('url', '')).netloc}"


def build_job_text_from_sourced_jobs(
    jobs: list[dict[str, Any]],
    extra_job_details: str,
) -> str:
    chunks = []
    for job in jobs:
        chunks.append(
            "\n".join(
                [
                    f"Job source URL: {job.get('url', '')}",
                    f"Title: {job.get('title', '')}",
                    f"Company: {job.get('company', '')}",
                    f"Location: {job.get('location', '')}",
                    f"Seniority: {job.get('seniority', '')}",
                    f"Sourcing status: {job.get('status', 'Sourced')}",
                    f"Extraction status: {job.get('extraction_status', '')}",
                    f"Extracted job description: {job.get('description', '')}",
                    f"Sourcing notes: {job.get('notes', '')}",
                ]
            )
        )
    if extra_job_details.strip():
        chunks.append(extra_job_details.strip())
    return "\n\n---JOB---\n\n".join(chunk for chunk in chunks if chunk.strip())


def _job_source_id(url: str) -> str:
    return hashlib.sha1(url.strip().lower().encode("utf-8")).hexdigest()[:12]


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        slug = unquote(path_parts[-1]).replace("-", " ").replace("_", " ").strip()
        if slug:
            return slug[:80].title()
    return parsed.netloc or "Sourced job"


def fetch_job_page_text(url: str, max_chars: int = 30000) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 CareerPilot job sourcing bot",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=15) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        raw = response.read(800_000)
    html_text = raw.decode(charset, errors="replace")
    parser = _ReadableHtmlParser()
    parser.feed(html_text)
    return parser.title.strip(), parser.visible_text(max_chars)


def _apply_extracted_job_to_source(
    job: dict[str, Any],
    posting: JobPosting,
    page_title: str,
) -> None:
    job["title"] = posting.title or page_title or job.get("title", "")
    job["company"] = posting.company
    job["location"] = posting.location
    job["seniority"] = posting.seniority
    job["description"] = _job_description_from_posting(posting)
    job["extraction_status"] = "Extracted"


def _job_description_from_posting(posting: JobPosting) -> str:
    lines = [
        posting.raw_summary,
        _join_labeled_list("Required skills", posting.required_skills),
        _join_labeled_list("Preferred skills", posting.preferred_skills),
        _join_labeled_list("Responsibilities", posting.responsibilities),
        _join_labeled_list("Language requirements", posting.language_requirements),
        _join_labeled_list("Visa / Blue Card signals", posting.visa_blue_card_signals),
        _join_labeled_list(
            "Risks for international graduate",
            posting.risks_for_international_graduate,
        ),
    ]
    return "\n".join(line for line in lines if line).strip()


def _join_labeled_list(label: str, values: list[str]) -> str:
    return f"{label}: {', '.join(values)}" if values else ""


class _ReadableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._title_parts).split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "div", "li", "br", "section", "article", "h1", "h2", "h3"}:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "div", "li", "section", "article", "h1", "h2", "h3"}:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        if self._skip_depth == 0 and not self._in_title:
            self._text_parts.append(text)

    def visible_text(self, max_chars: int) -> str:
        text = "\n".join(
            line.strip()
            for line in " ".join(self._text_parts).splitlines()
            if line.strip()
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        if not text:
            raise ValueError("No readable page text found.")
        return text[:max_chars]


def render_candidate_input_section() -> str:
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
    st.session_state["latest_candidate_input_hash"] = (
        _candidate_input_hash(candidate_text) if candidate_text.strip() else None
    )
    return candidate_text


def run_initial_ranking(
    profile: CandidateProfile,
    job_text: str,
    model_name: str,
    top_n: int,
) -> None:
    run_id = st.session_state.get("run_id") or new_run_id()
    events: list[TraceEvent] = st.session_state.get("trace_events") or []
    st.session_state["run_id"] = run_id
    st.session_state["trace_events"] = events
    st.session_state["reranking"] = None
    st.session_state["feedback"] = None
    st.session_state["policy"] = None

    agent = CareerPilotAgent(model_name=model_name)
    rubric = default_rubric()

    try:
        with st.status("Running initial agent pass", expanded=True) as status:
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


def render_profile_generation_section(
    *,
    candidate_text: str,
    model_name: str,
    profile_definition_json: str,
    profile_definition_error: str | None,
    profile_is_current: bool,
) -> None:
    st.subheader("Candidate profile")
    profile: CandidateProfile | None = st.session_state.get("profile")

    if not candidate_text.strip():
        st.info("Upload a CV or paste fallback CV text to generate a candidate profile.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        generate = st.button(
            "Generate profile",
            type="primary",
            disabled=profile_definition_error is not None,
        )
    with col2:
        if profile is not None and profile_is_current:
            st.success("Profile is ready for ranking.")
        elif profile is not None:
            st.warning("Candidate input changed. Regenerate the profile before ranking.")
        else:
            st.caption("The profile is generated from CV text and extra notes.")

    profile_error = st.session_state.get("profile_error")
    if profile_error:
        st.error(profile_error)

    if generate:
        if generate_candidate_profile(candidate_text, model_name, profile_definition_json):
            st.rerun()

    if profile is not None and profile_is_current:
        render_profile_summary(profile, expanded=True)
    elif profile is not None:
        render_profile_summary(profile, expanded=False)


def generate_candidate_profile(
    candidate_text: str,
    model_name: str,
    profile_definition_json: str,
) -> bool:
    run_id = new_run_id()
    events: list[TraceEvent] = []
    st.session_state["run_id"] = run_id
    st.session_state["trace_events"] = events
    st.session_state["jobs"] = None
    st.session_state["initial_ranking"] = None
    st.session_state["evaluation"] = None
    st.session_state["feedback"] = None
    st.session_state["policy"] = None
    st.session_state["reranking"] = None
    st.session_state["profile_definition_json"] = profile_definition_json
    st.session_state["profile_error"] = None

    agent = CareerPilotAgent(model_name=model_name)
    try:
        with st.status("Generating candidate profile", expanded=True) as status:
            st.write("Reading CV and notes")
            profile = agent.extract_profile(candidate_text, profile_definition_json)
            trace_event(
                events,
                run_id,
                "profile_extraction",
                "Candidate profile generated from CV and notes.",
                {
                    "target_roles": profile.target_roles,
                    "visa_status": profile.visa_status,
                    "controlled_metrics": [
                        metric.key for metric in profile.controlled_metrics
                    ],
                },
            )
            status.update(label="Candidate profile ready", state="complete")

        st.session_state["profile"] = profile
        st.session_state["candidate_input_hash"] = _candidate_input_hash(candidate_text)
        return True
    except Exception as exc:
        error = f"Profile generation failed: {exc}"
        st.session_state["profile_error"] = error
        trace_event(events, run_id, "profile_extraction_error", str(exc), status="error")
        st.error(error)
        return False
    finally:
        agent.close()


def _load_profile_definition_json() -> tuple[str | None, str | None]:
    profile_definition_text = _load_profile_definition_text()

    profile_definition, error = _parse_profile_definition(profile_definition_text)
    if error:
        return None, f"Profile metrics configuration is invalid: {error}"

    enabled_fields = _enabled_profile_fields(profile_definition)
    if not enabled_fields:
        return None, "Profile metrics configuration must keep at least one field enabled."

    return json.dumps(profile_definition, indent=2), None


def _load_profile_definition_text() -> str:
    try:
        return PROFILE_DEFINITION_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return json.dumps({"version": "custom", "fields": []}, indent=2)


def _parse_profile_definition(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        profile_definition = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"profile.json is invalid JSON: {exc.msg} at line {exc.lineno}."

    if not isinstance(profile_definition, dict):
        return {}, "profile.json must be a JSON object."

    fields = _profile_definition_fields(profile_definition)
    if not isinstance(fields, list):
        return {}, "profile.json must include a fields array."
    if not fields:
        return {}, "profile.json must define at least one field."

    seen_keys: set[str] = set()
    for index, field in enumerate(fields, start=1):
        if not isinstance(field, dict):
            return {}, f"profile.json field #{index} must be an object."
        key = str(field.get("key", "")).strip()
        if not key:
            return {}, f"profile.json field #{index} is missing key."
        if key in seen_keys:
            return {}, f"profile.json field key '{key}' is duplicated."
        seen_keys.add(key)
        field["key"] = key
        field.setdefault("label", key.replace("_", " ").title())
        field.setdefault("enabled", True)
        field.setdefault("required", False)
        field.setdefault("value_type", "string")

    return profile_definition, None


def _profile_definition_fields(profile_definition: dict[str, Any]) -> list[Any] | None:
    fields = profile_definition.get("fields")
    if fields is None:
        fields = profile_definition.get("dimensions")
    return fields


def _enabled_profile_fields(profile_definition: dict[str, Any]) -> list[dict[str, Any]]:
    fields = _profile_definition_fields(profile_definition) or []
    return [
        field
        for field in fields
        if isinstance(field, dict) and field.get("enabled", True) is not False
    ]


def _candidate_input_hash(candidate_text: str) -> str:
    return hashlib.sha256(candidate_text.strip().encode("utf-8")).hexdigest()


def _profile_is_current(candidate_text: str) -> bool:
    if not candidate_text.strip() or st.session_state.get("profile") is None:
        return False
    return st.session_state.get("candidate_input_hash") == _candidate_input_hash(candidate_text)


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


def render_initial_section(top_n: int, include_profile: bool = True) -> None:
    profile: CandidateProfile = st.session_state["profile"]
    ranking: RankingResult = st.session_state["initial_ranking"]
    evaluation: EvaluationReport | None = st.session_state.get("evaluation")

    st.header("Initial ranking")
    st.caption("This is Gemini's first pass before your feedback constraints are applied.")
    if include_profile:
        render_profile_summary(profile, expanded=False)
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


def render_profile_summary(profile: CandidateProfile, expanded: bool) -> None:
    with st.expander("Generated profile", expanded=expanded):
        cols = st.columns(4)
        cols[0].metric("Experience", _format_years(profile.years_experience))
        cols[1].metric("Seniority", profile.seniority_target or profile.seniority_level or "Unclear")
        cols[2].metric("Locations", str(len(profile.locations)))
        cols[3].metric("Languages", str(len(profile.languages)))

        st.write("Target roles")
        _render_pills(profile.target_roles)

        st.write("Core skills")
        _render_pills(profile.core_skills)

        cols = st.columns(2)
        with cols[0]:
            st.write("Visa / Blue Card")
            st.write(profile.visa_status or profile.blue_card_relevance or "Not clear yet")
            if profile.blue_card_relevance:
                st.caption(profile.blue_card_relevance)

            st.write("Location flexibility")
            st.write(profile.location_flexibility or ", ".join(profile.locations) or "Not clear yet")

        with cols[1]:
            st.write("Languages")
            if profile.languages:
                for language in profile.languages:
                    st.markdown(
                        f"- **{language.language}**: {language.level}"
                        + (f" ({language.evidence})" if language.evidence else "")
                    )
            else:
                st.caption("None reported")

        cols = st.columns(2)
        with cols[0]:
            _render_list("Hard constraints", profile.hard_constraints or profile.constraints)
        with cols[1]:
            _render_list("Soft preferences", profile.soft_preferences)

        _render_list("Uncertainties", profile.uncertainty_fields or profile.uncertainties)

        if profile.controlled_metrics:
            st.write("Profile details")
            for metric in profile.controlled_metrics:
                with st.container(border=True):
                    cols = st.columns([2, 1])
                    cols[0].write(metric.label or metric.key.replace("_", " ").title())
                    cols[1].caption(metric.confidence or "Confidence unclear")
                    st.write(_format_metric_value(metric.value) or "Not clear yet")
                    if metric.uncertainty:
                        st.caption(f"Uncertainty: {metric.uncertainty}")
                    if metric.evidence:
                        st.caption("Evidence: " + "; ".join(metric.evidence))


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
                _render_metadata(event.metadata)


def _render_metadata(metadata: dict[str, Any]) -> None:
    rows = [
        {"Name": key.replace("_", " ").title(), "Value": _format_metadata_value(value)}
        for key, value in metadata.items()
    ]
    st.table(rows)


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_format_metadata_value(item) for item in value) or "None"
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {_format_metadata_value(item)}" for key, item in value.items()
        )
    if value is None or value == "":
        return "None"
    return str(value)


def _render_list(label: str, values: list[str]) -> None:
    st.write(label)
    if values:
        for value in values:
            st.markdown(f"- {value}")
    else:
        st.caption("None reported")


def _render_pills(values: list[str]) -> None:
    if not values:
        st.caption("None reported")
        return
    pill_text = " ".join(f"`{value}`" for value in values)
    st.markdown(pill_text)


def _format_years(years: float | None) -> str:
    if years is None:
        return "Unclear"
    if years == int(years):
        return f"{int(years)} yr"
    return f"{years:.1f} yrs"


def _format_metric_value(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if not value.startswith(("[", "{")):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return _format_metadata_value(parsed)


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
        "sourced_jobs": [],
        "job_pool": [],
        "ai_job_search_results": [],
        "ai_job_search_queries": [],
        "profile_definition_json": None,
        "candidate_input_hash": None,
        "profile_error": None,
        "top_n": 5,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_state() -> None:
    list_state_keys = {
        "trace_events",
        "sourced_jobs",
        "job_pool",
        "ai_job_search_results",
        "ai_job_search_queries",
    }
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
        "sourced_jobs",
        "job_pool",
        "ai_job_search_results",
        "ai_job_search_queries",
        "profile_definition_json",
        "candidate_input_hash",
        "profile_error",
    ]:
        st.session_state[key] = [] if key in list_state_keys else None


if __name__ == "__main__":
    main()
