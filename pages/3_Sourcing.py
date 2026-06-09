"""Job pool building step for CareerPilot."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import streamlit as st

import app as careerpilot
from agent import CareerPilotAgent, JobPosting


JOB_POOL_COLUMNS = [
    "selected",
    "title",
    "company",
    "location",
    "seniority",
    "source",
    "parse_status",
]


def init_job_pool() -> list[dict[str, Any]]:
    """Initialize the central job pool, migrating older sourced jobs when present."""
    legacy_jobs = st.session_state.get("sourced_jobs") or []
    should_migrate_legacy_jobs = (
        "job_pool" not in st.session_state
        or st.session_state["job_pool"] is None
        or (not st.session_state["job_pool"] and legacy_jobs)
    )
    if should_migrate_legacy_jobs:
        st.session_state["job_pool"] = [_job_from_legacy_source(job) for job in legacy_jobs]

    pool: list[dict[str, Any]] = st.session_state.setdefault("job_pool", [])
    for job in pool:
        _normalize_job_record(job)
    return pool


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip().rstrip(").,;]"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""

    normalized = ParseResult(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/"),
        params="",
        query=parsed.query,
        fragment="",
    )
    return urlunparse(normalized)


def make_job_id(value: str) -> str:
    return hashlib.sha1(value.strip().lower().encode("utf-8")).hexdigest()[:12]


def add_links_to_job_pool(links: list[str], source_label: str) -> tuple[int, int]:
    pool = init_job_pool()
    existing_urls = {normalize_url(job.get("url", "")) for job in pool if job.get("url")}
    existing_ids = {job["id"] for job in pool}
    added = 0
    duplicates = 0

    for raw_link in links:
        url = normalize_url(raw_link)
        if not url:
            continue
        job_id = make_job_id(url)
        if url in existing_urls or job_id in existing_ids:
            duplicates += 1
            continue

        pool.append(_new_job_record(job_id=job_id, url=url, source=source_label))
        existing_urls.add(url)
        existing_ids.add(job_id)
        added += 1

    st.session_state["job_pool"] = pool
    sync_job_pool_to_sourced_jobs()
    return added, duplicates


def get_pending_jobs() -> list[dict[str, Any]]:
    return [job for job in init_job_pool() if job.get("parse_status") == "pending"]


def parse_pending_jobs(model_name: str) -> tuple[int, int]:
    pending_jobs = get_pending_jobs()
    if not pending_jobs:
        return 0, 0

    agent = CareerPilotAgent(model_name=model_name)
    updated = 0
    failed = 0
    progress = st.progress(0)

    try:
        with st.status("Parsing pending jobs", expanded=True) as status:
            for index, job in enumerate(pending_jobs, start=1):
                label = job.get("url") or job.get("file_name") or job.get("title") or job["id"]
                st.write(f"Parsing {label}")
                try:
                    posting, page_title = _extract_posting_for_job(agent, job)
                    _apply_parsed_posting(job, posting, page_title)
                    updated += 1
                except Exception as exc:
                    job["parse_status"] = "failed"
                    job["parse_error"] = str(exc)
                    job["extraction_status"] = f"Needs manual details: {exc}"
                    failed += 1
                progress.progress(index / len(pending_jobs))

            st.session_state["job_pool"] = init_job_pool()
            sync_job_pool_to_sourced_jobs()
            status.update(
                label=f"Parsing finished: {updated} parsed, {failed} failed",
                state="complete" if failed == 0 else "error",
            )
    finally:
        agent.close()

    return updated, failed


def prepare_jobs_for_ranking(job_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranking_jobs: list[dict[str, Any]] = []
    for job in job_pool:
        if job.get("parse_status") != "parsed":
            continue

        ranking_jobs.append(
            {
                "id": job["id"],
                "url": job.get("url", ""),
                "title": job.get("title") or _title_from_url(job.get("url", "")),
                "company": job.get("company") or "",
                "location": job.get("location") or "",
                "seniority": job.get("seniority") or "",
                "description": job.get("description") or "",
                "source": job.get("source") or "",
                "status": "Selected for ranking" if job.get("selected", True) else "Sourced",
                "extraction_status": "Extracted",
                "notes": job.get("notes") or "",
                "rank": bool(job.get("selected", True)),
                "requirements": job.get("requirements") or [],
            }
        )
    return ranking_jobs


def sync_job_pool_to_sourced_jobs() -> None:
    st.session_state["sourced_jobs"] = prepare_jobs_for_ranking(init_job_pool())


def import_files_to_job_pool(uploaded_files: list[Any], source_label: str) -> tuple[int, int, list[str]]:
    added = 0
    duplicates = 0
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            file_text = _read_uploaded_job_file(uploaded_file)
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")
            continue

        urls = careerpilot.parse_job_urls(file_text)
        if urls:
            new_added, new_duplicates = add_links_to_job_pool(
                urls,
                f"{source_label}: {uploaded_file.name}",
            )
            added += new_added
            duplicates += new_duplicates
            continue

        # TODO: Split multi-job files into individual postings before adding them.
        # For the MVP, a file without URLs becomes one pending job backed by raw text.
        was_added = _add_file_placeholder(uploaded_file.name, file_text, source_label)
        if was_added:
            added += 1
        else:
            duplicates += 1

    sync_job_pool_to_sourced_jobs()
    return added, duplicates, errors


def generate_job_search_queries(candidate_profile: dict, search_preferences: dict) -> list[str]:
    target_role = _first_non_empty(
        search_preferences.get("target_role"),
        *_as_list(candidate_profile.get("target_roles")),
        "software engineer",
    )
    location = _first_non_empty(
        search_preferences.get("location"),
        *_as_list(candidate_profile.get("locations")),
        "Germany",
    )
    seniority = _first_non_empty(
        search_preferences.get("seniority"),
        candidate_profile.get("seniority_target"),
        candidate_profile.get("seniority_level"),
    )
    keywords = _dedupe_preserve_order(
        _split_keywords(search_preferences.get("keywords", ""))
        + _as_list(candidate_profile.get("core_skills"))[:4]
    )

    keyword_text = " ".join(keywords[:5])
    parts = [target_role, location, seniority, keyword_text]
    base_query = " ".join(str(part).strip() for part in parts if str(part).strip())

    queries = [
        f'{base_query} jobs',
        f'{base_query} careers',
        f'site:greenhouse.io {base_query}',
        f'site:lever.co {base_query}',
        f'site:ashbyhq.com {base_query}',
        f'site:workdayjobs.com {base_query}',
    ]
    return _dedupe_preserve_order(queries)


def search_job_links(
    candidate_profile: dict,
    search_preferences: dict,
    num_results: int,
) -> tuple[list[dict[str, str]], list[str]]:
    queries = generate_job_search_queries(candidate_profile, search_preferences)
    provider = _search_provider()
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for query in queries:
        if len(results) >= num_results:
            break
        per_query_limit = min(10, max(3, num_results - len(results)))
        for result in _call_search_provider(provider, query, per_query_limit):
            url = normalize_url(result.get("url", ""))
            if not url or url in seen_urls:
                continue
            if not _is_probable_job_result(url, result.get("title", ""), result.get("snippet", "")):
                continue

            seen_urls.add(url)
            results.append(
                {
                    "title": result.get("title", "")[:160],
                    "url": url,
                    "snippet": result.get("snippet", "")[:500],
                    "source": f"AI search ({provider})",
                    "query": query,
                }
            )
            if len(results) >= num_results:
                break

    return results, queries


def render_job_pool_table() -> None:
    pool = init_job_pool()
    st.subheader("Existing Job Pool")
    if not pool:
        st.info("Your job pool is empty. Add jobs below to start ranking.")
        return

    rows = [
        {
            "id": job["id"],
            "selected": bool(job.get("selected", True)),
            "title": job.get("title") or "",
            "company": job.get("company") or "",
            "location": job.get("location") or "",
            "seniority": job.get("seniority") or "",
            "source": job.get("source") or "",
            "parse_status": job.get("parse_status") or "pending",
        }
        for job in pool
    ]

    edited_rows = st.data_editor(
        rows,
        hide_index=True,
        use_container_width=True,
        column_order=JOB_POOL_COLUMNS,
        column_config={
            "selected": st.column_config.CheckboxColumn("Selected", help="Include by default on the Ranking page."),
            "title": st.column_config.TextColumn("Title"),
            "company": st.column_config.TextColumn("Company"),
            "location": st.column_config.TextColumn("Location"),
            "seniority": st.column_config.TextColumn("Seniority"),
            "source": st.column_config.TextColumn("Source"),
            "parse_status": st.column_config.TextColumn("Parse status"),
        },
        disabled=["source", "parse_status"],
        key="job_pool_editor",
    )
    _apply_job_pool_edits(edited_rows)
    sync_job_pool_to_sourced_jobs()


def render_add_jobs_tabs() -> None:
    st.subheader("Add Jobs")
    paste_tab, upload_tab, search_tab = st.tabs(["Paste Links", "Upload Files", "Search with AI"])

    with paste_tab:
        pasted_links = st.text_area(
            "Job URLs",
            height=180,
            placeholder="Paste one job link per line.",
            key="job_pool_pasted_links",
        )
        source_label = st.text_input(
            "Source label",
            value="Manual links",
            key="job_pool_link_source",
        )
        if st.button("Import links", type="primary"):
            links = _links_from_pasted_lines(pasted_links)
            if not links:
                st.warning("Paste at least one valid http or https job link.")
            else:
                added, duplicates = add_links_to_job_pool(
                    links,
                    source_label.strip() or "Manual links",
                )
                st.success(f"Imported {added} links. Skipped {duplicates} duplicates.")
                st.rerun()

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Upload job files",
            type=["txt", "csv", "html", "pdf"],
            accept_multiple_files=True,
            help="TXT, CSV, HTML, and text-based PDF files are supported.",
        )
        file_source_label = st.text_input(
            "Source label",
            value="Uploaded files",
            key="job_pool_file_source",
        )
        if st.button("Import from files"):
            if not uploaded_files:
                st.warning("Upload at least one file.")
            else:
                added, duplicates, errors = import_files_to_job_pool(
                    uploaded_files,
                    file_source_label.strip() or "Uploaded files",
                )
                st.success(f"Imported {added} jobs. Skipped {duplicates} duplicates.")
                for error in errors:
                    st.warning(error)
                st.rerun()

    with search_tab:
        target_role = st.text_input("Target role", key="job_search_target_role")
        location = st.text_input("Location", key="job_search_location")
        seniority = st.text_input("Seniority", key="job_search_seniority")
        keywords = st.text_input("Keywords", key="job_search_keywords")
        num_jobs_to_search = st.number_input(
            "How many jobs do you want to search for?",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
        )

        # TODO: Replace direct web-search API calls with MCP connectors when the
        # app needs authenticated job-board, ATS, or enterprise search tools.
        if st.button("Search jobs"):
            search_preferences = {
                "target_role": target_role,
                "location": location,
                "seniority": seniority,
                "keywords": keywords,
            }
            try:
                with st.status("Searching for candidate job links", expanded=True) as status:
                    results, queries = search_job_links(
                        _candidate_profile_to_dict(st.session_state.get("profile")),
                        search_preferences,
                        int(num_jobs_to_search),
                    )
                    st.session_state["ai_job_search_results"] = results
                    st.session_state["ai_job_search_queries"] = queries
                    status.update(
                        label=f"Found {len(results)} candidate job links",
                        state="complete",
                    )
            except Exception as exc:
                st.session_state["ai_job_search_results"] = []
                st.session_state["ai_job_search_queries"] = []
                st.error(str(exc))

        _render_ai_search_results()


def render_parse_jobs_section(model_name: str) -> None:
    st.subheader("Parse Job Details")
    pending_count = len(get_pending_jobs())
    parsed_count = len([job for job in init_job_pool() if job.get("parse_status") == "parsed"])
    failed_count = len([job for job in init_job_pool() if job.get("parse_status") == "failed"])

    st.caption(f"{pending_count} pending, {parsed_count} parsed, {failed_count} failed.")
    if st.button("Parse pending jobs", type="primary", disabled=pending_count == 0):
        updated, failed = parse_pending_jobs(model_name)
        if updated:
            st.success(f"Parsed {updated} jobs.")
        if failed:
            st.warning(f"{failed} jobs failed to parse.")
        st.rerun()


def render_continue_to_ranking() -> None:
    st.subheader("Continue to Ranking")
    parsed_jobs = [job for job in init_job_pool() if job.get("parse_status") == "parsed"]
    if not parsed_jobs:
        st.info("Parse at least one job before ranking.")
        st.button("Continue to Ranking", disabled=True)
        return

    sync_job_pool_to_sourced_jobs()
    if st.button("Continue to Ranking", type="primary"):
        try:
            st.switch_page("pages/4_Ranking.py")
        except Exception:
            st.warning("Use the sidebar page navigation for the Ranking step.")


def _render_ai_search_results() -> None:
    results: list[dict[str, str]] = st.session_state.get("ai_job_search_results") or []
    queries: list[str] = st.session_state.get("ai_job_search_queries") or []
    if not results and not queries:
        return

    if queries:
        with st.expander("Search queries used", expanded=False):
            for query in queries:
                st.write(query)

    if not results:
        st.info("No candidate job links found. Try a broader role, location, or keyword set.")
        return

    rows = [
        {
            "add": True,
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
            "query": result.get("query", ""),
        }
        for result in results
    ]
    edited_rows = st.data_editor(
        rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "add": st.column_config.CheckboxColumn("Add"),
            "title": st.column_config.TextColumn("Title"),
            "url": st.column_config.LinkColumn("URL"),
            "snippet": st.column_config.TextColumn("Snippet"),
            "query": st.column_config.TextColumn("Query"),
        },
        disabled=["title", "url", "snippet", "query"],
        key="ai_job_search_results_editor",
    )
    selected_urls = [row["url"] for row in edited_rows if row.get("add") and row.get("url")]
    if st.button("Add selected jobs to pool", disabled=not selected_urls):
        added, duplicates = add_links_to_job_pool(selected_urls, "AI search")
        st.success(f"Added {added} jobs to the pool. Skipped {duplicates} duplicates.")
        st.rerun()


def _call_search_provider(provider: str, query: str, max_results: int) -> list[dict[str, str]]:
    if provider == "serper":
        return _search_serper(query, max_results)
    if provider == "tavily":
        return _search_tavily(query, max_results)
    if provider == "brave":
        return _search_brave(query, max_results)
    if provider == "google_cse":
        return _search_google_cse(query, max_results)
    raise ValueError(
        "Unsupported JOB_SEARCH_PROVIDER. Use serper, tavily, brave, or google_cse."
    )


def _search_serper(query: str, max_results: int) -> list[dict[str, str]]:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set SERPER_API_KEY, or configure JOB_SEARCH_PROVIDER with Tavily, Brave, or Google CSE keys."
        )

    payload = json.dumps({"q": query, "num": max_results}).encode("utf-8")
    request = Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    data = _load_json_response(request)
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in data.get("organic", [])[:max_results]
    ]


def _search_tavily(query: str, max_results: int) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Set TAVILY_API_KEY to use JOB_SEARCH_PROVIDER=tavily.")

    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }
    ).encode("utf-8")
    request = Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    data = _load_json_response(request)
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
        }
        for item in data.get("results", [])[:max_results]
    ]


def _search_brave(query: str, max_results: int) -> list[dict[str, str]]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("Set BRAVE_SEARCH_API_KEY to use JOB_SEARCH_PROVIDER=brave.")

    params = urlencode({"q": query, "count": max_results})
    request = Request(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
    )
    data = _load_json_response(request)
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        }
        for item in data.get("web", {}).get("results", [])[:max_results]
    ]


def _search_google_cse(query: str, max_results: int) -> list[dict[str, str]]:
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    engine_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not engine_id:
        raise RuntimeError(
            "Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID to use JOB_SEARCH_PROVIDER=google_cse."
        )

    params = urlencode(
        {
            "key": api_key,
            "cx": engine_id,
            "q": query,
            "num": min(max_results, 10),
        }
    )
    request = Request(f"https://www.googleapis.com/customsearch/v1?{params}")
    data = _load_json_response(request)
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in data.get("items", [])[:max_results]
    ]


def _load_json_response(request: Request) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read(1_000_000)
    except HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Search API request failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Search API request failed: {exc.reason}") from exc

    return json.loads(raw.decode("utf-8", errors="replace"))


def _search_provider() -> str:
    configured_provider = os.getenv("JOB_SEARCH_PROVIDER", "").strip().lower()
    if configured_provider:
        return configured_provider
    if os.getenv("SERPER_API_KEY"):
        return "serper"
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        return "brave"
    if os.getenv("GOOGLE_CSE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        return "google_cse"
    return "serper"


def _is_probable_job_result(url: str, title: str, snippet: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    blocked_hosts = {
        "google.com",
        "www.google.com",
        "bing.com",
        "www.bing.com",
        "duckduckgo.com",
        "www.duckduckgo.com",
        "facebook.com",
        "www.facebook.com",
        "youtube.com",
        "www.youtube.com",
    }
    if host in blocked_hosts:
        return False

    combined = " ".join([url, title, snippet]).lower()
    job_signals = [
        "job",
        "jobs",
        "career",
        "careers",
        "opening",
        "apply",
        "greenhouse",
        "lever",
        "ashby",
        "workday",
        "smartrecruiters",
        "personio",
        "bamboohr",
    ]
    return any(signal in combined for signal in job_signals)


def _new_job_record(job_id: str, url: str, source: str) -> dict[str, Any]:
    return {
        "id": job_id,
        "url": url,
        "title": None,
        "company": None,
        "location": None,
        "seniority": None,
        "description": None,
        "requirements": [],
        "source": source,
        "parse_status": "pending",
        "selected": True,
        "notes": "",
    }


def _normalize_job_record(job: dict[str, Any]) -> None:
    url = normalize_url(job.get("url", ""))
    job["url"] = url
    job["id"] = job.get("id") or make_job_id(url or job.get("file_name", ""))
    job.setdefault("title", None)
    job.setdefault("company", None)
    job.setdefault("location", None)
    job.setdefault("seniority", None)
    job.setdefault("description", None)
    job.setdefault("requirements", [])
    job.setdefault("source", "")
    job.setdefault("parse_status", "pending")
    job.setdefault("selected", bool(job.get("rank", True)))
    job.setdefault("notes", "")


def _job_from_legacy_source(job: dict[str, Any]) -> dict[str, Any]:
    extraction_status = job.get("extraction_status", "Not extracted")
    if extraction_status == "Extracted":
        parse_status = "parsed"
    elif str(extraction_status).startswith("Needs manual details"):
        parse_status = "failed"
    else:
        parse_status = "pending"

    migrated = {
        "id": job.get("id") or make_job_id(job.get("url", "")),
        "url": job.get("url", ""),
        "title": job.get("title") or None,
        "company": job.get("company") or None,
        "location": job.get("location") or None,
        "seniority": job.get("seniority") or None,
        "description": job.get("description") or None,
        "requirements": job.get("requirements") or [],
        "source": job.get("source") or "",
        "parse_status": parse_status,
        "selected": bool(job.get("rank", True)),
        "notes": job.get("notes") or "",
    }
    _normalize_job_record(migrated)
    return migrated


def _links_from_pasted_lines(pasted_links: str) -> list[str]:
    links: list[str] = []
    for line in pasted_links.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed_links = careerpilot.parse_job_urls(line)
        links.extend(parsed_links or [line])
    return links


def _candidate_profile_to_dict(candidate_profile: Any) -> dict[str, Any]:
    if candidate_profile is None:
        return {}
    if hasattr(candidate_profile, "model_dump"):
        return candidate_profile.model_dump()
    if isinstance(candidate_profile, dict):
        return candidate_profile
    return {}


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _split_keywords(value: str) -> list[str]:
    return [
        keyword.strip()
        for keyword in value.replace("\n", ",").split(",")
        if keyword.strip()
    ]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _apply_job_pool_edits(edited_rows: list[dict[str, Any]]) -> None:
    pool = init_job_pool()
    by_id = {job["id"]: job for job in pool}
    for row in edited_rows:
        job = by_id.get(row["id"])
        if not job:
            continue
        job["selected"] = bool(row["selected"])
        job["title"] = _empty_to_none(row["title"])
        job["company"] = _empty_to_none(row["company"])
        job["location"] = _empty_to_none(row["location"])
        job["seniority"] = _empty_to_none(row["seniority"])
    st.session_state["job_pool"] = pool


def _read_uploaded_job_file(uploaded_file: Any) -> str:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    data = uploaded_file.getvalue()
    if suffix == "pdf":
        return careerpilot._extract_pdf_text(data)

    text = data.decode("utf-8", errors="ignore")
    if suffix in {"html", "htm"}:
        parser = careerpilot._ReadableHtmlParser()
        parser.feed(text)
        return parser.visible_text(50000)
    return text


def _add_file_placeholder(file_name: str, file_text: str, source_label: str) -> bool:
    pool = init_job_pool()
    content_hash = make_job_id(f"{file_name}:{file_text[:5000]}")
    job_id = make_job_id(f"file:{content_hash}")
    if any(job["id"] == job_id for job in pool):
        return False

    pool.append(
        {
            "id": job_id,
            "url": "",
            "title": file_name,
            "company": None,
            "location": None,
            "seniority": None,
            "description": None,
            "requirements": [],
            "source": source_label,
            "parse_status": "pending",
            "selected": True,
            "notes": "",
            "file_name": file_name,
            "raw_text": file_text,
        }
    )
    st.session_state["job_pool"] = pool
    return True


def _extract_posting_for_job(
    agent: CareerPilotAgent,
    job: dict[str, Any],
) -> tuple[JobPosting, str]:
    raw_text = job.get("raw_text") or ""
    page_title = job.get("title") or ""
    if raw_text:
        extraction_text = "\n\n".join(
            [
                f"Job source file: {job.get('file_name', '')}",
                "Extract one job posting from the file text below.",
                raw_text,
            ]
        )
    else:
        url = job.get("url", "")
        if not url:
            raise ValueError("Job has no URL or uploaded text to parse.")
        page_title, page_text = careerpilot.fetch_job_page_text(url)
        extraction_text = "\n\n".join(
            [
                f"Job source URL: {url}",
                f"Page title: {page_title}",
                "Extract one job posting from the page text below.",
                page_text,
            ]
        )

    extracted_jobs = agent.extract_jobs(extraction_text)
    if not extracted_jobs:
        raise RuntimeError("No job posting was extracted.")
    return extracted_jobs[0], page_title


def _apply_parsed_posting(job: dict[str, Any], posting: JobPosting, page_title: str) -> None:
    job["title"] = posting.title or page_title or job.get("title")
    job["company"] = posting.company or None
    job["location"] = posting.location or None
    job["seniority"] = posting.seniority or None
    job["description"] = careerpilot._job_description_from_posting(posting)
    job["requirements"] = posting.required_skills
    job["remote_policy"] = posting.remote_policy
    job["minimum_years_experience"] = posting.minimum_years_experience
    job["language_requirements"] = posting.language_requirements
    job["parse_status"] = "parsed"
    job["parse_error"] = ""
    job["extraction_status"] = "Extracted"


def _title_from_url(url: str) -> str:
    if not url:
        return "Uploaded job"
    return careerpilot.sourced_job_label({"url": url, "title": ""}).split(" - ", 1)[0]


def _empty_to_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


model_name, _ = careerpilot.setup_page("Job Pool - CareerPilot")
init_job_pool()
sync_job_pool_to_sourced_jobs()

st.title("Build Job Pool")
st.caption("Add jobs from links, files, or AI search. We will parse and structure them for ranking.")

render_job_pool_table()
render_add_jobs_tabs()
render_parse_jobs_section(model_name)
render_continue_to_ranking()
