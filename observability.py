"""Local trace hooks with optional Phoenix/Arize tracing."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    run_id: str
    step: str
    status: str = "ok"
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


_provider: Any | None = None
_setup_attempted = False


def new_run_id() -> str:
    return uuid.uuid4().hex[:10]


def setup_observability() -> Any | None:
    """Initialize Phoenix tracing when configured.

    The app remains fully runnable without Phoenix dependencies or credentials.
    """

    global _provider, _setup_attempted
    if _provider is not None or _setup_attempted:
        return _provider

    _setup_attempted = True
    api_key = os.getenv("PHOENIX_API_KEY", "").strip()
    endpoint = _phoenix_endpoint()
    if not api_key or not endpoint:
        return None

    try:
        from phoenix.otel import register

        _provider = register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "careerpilot"),
            batch=False,
            auto_instrument=True,
            verbose=False,
        )
    except Exception:
        _provider = None
    return _provider


def trace_event(
    events: list[TraceEvent],
    run_id: str,
    step: str,
    message: str = "",
    metadata: dict[str, Any] | None = None,
    status: str = "ok",
) -> TraceEvent:
    """Record one local trace event and attempt optional Phoenix export."""

    event = TraceEvent(
        run_id=run_id,
        step=step,
        status=status,
        message=message,
        metadata=metadata or {},
    )
    events.append(event)
    _record_phoenix_span(event)
    return event


def record_human_feedback(
    events: list[TraceEvent],
    run_id: str,
    feedback: dict[str, Any],
    policy: dict[str, Any],
) -> TraceEvent:
    return trace_event(
        events=events,
        run_id=run_id,
        step="human_feedback",
        message="User feedback converted into ranking constraints.",
        metadata={"feedback": feedback, "policy": policy},
    )


def get_previous_feedback_for_session(session_id: str = "") -> list[dict[str, Any]]:
    """Retrieve prior run history from Phoenix via the MCP server.

    Replaces the old hard-coded `[]`. The application runtime now spawns the
    Phoenix MCP server and reads recent evaluation / feedback / improvement spans
    so the agent can learn across runs. Returns `[]` only when Phoenix is
    unconfigured or unreachable (graceful degradation).
    """

    _ = session_id
    try:
        import mcp_client

        return mcp_client.recent_eval_spans(limit=25)
    except Exception:
        return []


def phoenix_status() -> str:
    endpoint = _phoenix_endpoint()
    if endpoint and os.getenv("PHOENIX_API_KEY"):
        if _provider is not None:
            return f"Phoenix tracing enabled: {endpoint}"
        return f"Phoenix configured, local fallback active until tracing packages load: {endpoint}"
    return "Local trace only. Phoenix/Arize is not configured."


def _record_phoenix_span(event: TraceEvent) -> None:
    if setup_observability() is None:
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("careerpilot")
        with tracer.start_as_current_span(event.step) as span:
            span.set_attribute("careerpilot.run_id", event.run_id)
            span.set_attribute("careerpilot.step", event.step)
            span.set_attribute("careerpilot.status", event.status)
            span.set_attribute("careerpilot.message", event.message)
            if event.metadata:
                span.set_attribute(
                    "careerpilot.metadata_json",
                    json.dumps(event.metadata, default=str)[:8000],
                )
    except Exception:
        return


def _phoenix_endpoint() -> str:
    return (
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or os.getenv("PHOENIX_BASE_URL")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.getenv("ARIZE_PHOENIX_ENDPOINT")
        or ""
    ).strip()
