"""Runtime client for the Arize Phoenix MCP server.

This is the integration that gives CareerPilot its "superpower" for the Arize
track: the *application runtime* (not just an external CLI config) spawns the
Phoenix MCP server over stdio and calls its tools to

  * read prior performance  -> get-spans            (replaces the old `[]` stub)
  * warm-start from memory   -> get-latest-prompt    (last learned rubric)
  * persist what it learned  -> upsert-prompt         (cross-run memory)

Every call degrades gracefully: if node/npx, the MCP server, or the Phoenix
space is unreachable, helpers log and return empty/None so the agent still runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Callable

# Name of the Phoenix prompt used as the agent's persistent rubric memory.
# Phoenix strips hyphens from prompt identifiers, so use underscores.
RUBRIC_MEMORY_PROMPT = os.getenv("PHOENIX_RUBRIC_PROMPT", "careerpilot_rubric")


def _base_url() -> str:
    return (
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or os.getenv("PHOENIX_BASE_URL")
        or os.getenv("ARIZE_PHOENIX_ENDPOINT")
        or "https://app.phoenix.arize.com"
    ).strip().strip("'\"")


def _api_key() -> str:
    return os.getenv("PHOENIX_API_KEY", "").strip()


def _project() -> str:
    return os.getenv("PHOENIX_PROJECT_NAME", "careerpilot").strip() or "careerpilot"


def is_configured() -> bool:
    """True when we have the minimum to reach a Phoenix space via MCP."""

    return bool(_api_key()) and bool(_base_url())


def _server_params() -> Any:
    """Build the stdio params for the Phoenix MCP server.

    The package spec is configurable so a container can point at a pre-installed
    global copy (`@arizeai/phoenix-mcp`, no tag) and skip the per-call registry
    check that `@latest` forces. Locally it defaults to `@latest`.
    """

    from mcp import StdioServerParameters

    package = os.getenv("PHOENIX_MCP_PACKAGE", "@arizeai/phoenix-mcp@latest")
    return StdioServerParameters(
        command=os.getenv("PHOENIX_MCP_COMMAND", "npx"),
        args=["-y", package, "--baseUrl", _base_url(), "--apiKey", _api_key()],
        env=os.environ.copy(),
    )


# --------------------------------------------------------------------------- #
# Async <-> sync bridge. Streamlit runs synchronously, so we execute every MCP
# session in a dedicated thread with its own event loop and join on it. Each
# call spawns a short-lived `npx @arizeai/phoenix-mcp` subprocess, runs the
# requested tool calls in one session, and tears it down.
# --------------------------------------------------------------------------- #


def _run_async(coro_factory: Callable[[], Any]) -> Any:
    result: dict[str, Any] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(coro_factory())
        except Exception as exc:  # noqa: BLE001 - surfaced to caller as error
            result["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=float(os.getenv("PHOENIX_MCP_TIMEOUT", "90")))
    if thread.is_alive():
        raise TimeoutError("Phoenix MCP call timed out.")
    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _call_tools(calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    """Open one MCP session and run a batch of (tool_name, args) calls."""

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _server_params()

    outputs: list[Any] = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool_name, args in calls:
                res = await session.call_tool(tool_name, args)
                outputs.append(_parse_content(res))
    return outputs


def _parse_content(result: Any) -> Any:
    """Flatten MCP tool result content into JSON (or raw text)."""

    texts: list[str] = []
    for chunk in getattr(result, "content", []) or []:
        text = getattr(chunk, "text", None)
        if text:
            texts.append(text)
    blob = "\n".join(texts).strip()
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return blob


def list_tools() -> list[str]:
    """Return the tool names the connected Phoenix MCP server exposes (debug)."""

    async def _go() -> list[str]:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        params = _server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [t.name for t in tools.tools]

    return _run_async(_go)


# --------------------------------------------------------------------------- #
# High-level memory helpers used by the orchestrator.
# --------------------------------------------------------------------------- #


def recent_eval_spans(limit: int = 25) -> list[dict[str, Any]]:
    """Read recent CareerPilot spans from Phoenix for cross-run learning.

    Returns a compact list of {step, message, output_gist} dicts drawn from the
    evaluator / ranking / feedback spans of past runs. Empty list on any failure
    (this is what replaces observability's old hard-coded `[]`).
    """

    if not is_configured():
        return []
    try:
        wanted = [
            "evaluator_result",
            "improved_reranking",
            "human_feedback",
            "self_improvement",
            "agent_stop",
        ]
        results = _run_async(
            lambda: _call_tools(
                [("get-spans", {"project_identifier": _project(), "limit": limit, "names": wanted})]
            )
        )
        payload = results[0] if results else None
        spans = (payload or {}).get("spans", []) if isinstance(payload, dict) else []
        compact: list[dict[str, Any]] = []
        for span in spans:
            attrs = span.get("attributes", {}) or {}
            compact.append(
                {
                    "step": span.get("name", ""),
                    "time": span.get("start_time", ""),
                    "input": _shorten(attrs.get("input.value", "")),
                    "output": _shorten(attrs.get("output.value", "")),
                }
            )
        return compact
    except Exception:
        return []


def read_rubric_memory() -> dict[str, Any] | None:
    """Fetch the last learned rubric stored as a Phoenix prompt, if any."""

    if not is_configured():
        return None
    try:
        results = _run_async(
            lambda: _call_tools(
                [("get-latest-prompt", {"prompt_identifier": RUBRIC_MEMORY_PROMPT})]
            )
        )
        payload = results[0] if results else None
        template = _extract_template_text(payload)
        if not template:
            return None
        blob = _first_json_object(template)
        return blob
    except Exception:
        return None


def write_rubric_memory(rubric_json: str, note: str = "") -> bool:
    """Persist the winning rubric as a new version of the memory prompt."""

    if not is_configured():
        return False
    try:
        description = (note or "CareerPilot self-improved ranking rubric.")[:280]
        _run_async(
            lambda: _call_tools(
                [
                    (
                        "upsert-prompt",
                        {
                            "name": RUBRIC_MEMORY_PROMPT,
                            "description": description,
                            "template": rubric_json,
                            "model_provider": "GOOGLE",
                            "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                            "temperature": 0.2,
                        },
                    )
                ]
            )
        )
        return True
    except Exception:
        return False


def _shorten(value: Any, limit: int = 600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def _extract_template_text(payload: Any) -> str:
    """Pull the template text out of a get-latest-prompt response.

    Phoenix stores prompts as chat templates, so the rubric JSON we wrote lives
    at template.messages[].content[].text. A 404 (prompt not found) comes back as
    a plain URL string, which we treat as "no memory".
    """

    if payload is None:
        return ""
    if isinstance(payload, str):
        # 404s arrive as "<url>: 404 Not Found".
        if payload.startswith("http") and "404" in payload:
            return ""
        return payload
    if not isinstance(payload, dict):
        return str(payload)

    template = payload.get("template")
    if isinstance(template, str) and template.strip():
        return template
    if isinstance(template, dict):
        texts: list[str] = []
        for message in template.get("messages", []) or []:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    text = part.get("text") if isinstance(part, dict) else None
                    if text:
                        texts.append(text)
        if texts:
            return "\n".join(texts)

    for key in ("content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : idx + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
