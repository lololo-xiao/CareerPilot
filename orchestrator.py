"""CareerPilot agent orchestrator.

This is the agentic core that turns the old straight-line pipeline into a
planning, self-improving, autonomously-stopping agent with persistent memory.

Backbone (deterministic, reliable to demo):

    recall memory (Phoenix MCP)  ->  plan  ->  [model-selected tools]
      ->  rank  ->  evaluate
      ->  while quality is low and budget remains:
              improve rubric  ->  re-rank  ->  re-evaluate     (self-improvement)
      ->  stop on threshold or max iterations                  (autonomous stop)
      ->  persist learned rubric to Phoenix (cross-run memory)

Each step appends an `AgentStep` so the UI can stream the agent's reasoning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import mcp_client
from agent import (
    CandidateProfile,
    CareerPilotAgent,
    JobPosting,
    RankingResult,
    RankingRubric,
    default_rubric,
)
from evaluators import (
    EvaluationReport,
    evaluate_ranking,
    generate_improved_rubric,
    needs_improvement,
)

StepSink = Callable[["AgentStep"], None]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


EVAL_THRESHOLD = _int_env("EVAL_THRESHOLD", 4)
MAX_IMPROVE_ITERS = _int_env("MAX_IMPROVE_ITERS", 2)


@dataclass
class AgentStep:
    kind: str  # recall | plan | tool | rank | evaluate | improve | stop | memory | error
    title: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | warn | error


@dataclass
class AgentIteration:
    index: int
    rubric_version: str
    ranking: RankingResult
    evaluation: EvaluationReport


@dataclass
class AgentRun:
    profile: CandidateProfile
    jobs: list[JobPosting]
    plan: str
    iterations: list[AgentIteration]
    final_ranking: RankingResult
    final_evaluation: EvaluationReport | None
    stop_reason: str
    memory_used: bool
    memory_written: bool
    steps: list[AgentStep]

    @property
    def improved(self) -> bool:
        return len(self.iterations) > 1


def _score_floor(report: EvaluationReport | None) -> int:
    if report is None:
        return 0
    return min(report.fit_quality, report.risk_detection, report.actionability)


def run_agent(
    profile: CandidateProfile,
    jobs: list[JobPosting],
    model_name: str,
    top_n: int = 5,
    on_step: StepSink | None = None,
) -> AgentRun:
    """Run the full agentic ranking loop and return a structured trace."""

    steps: list[AgentStep] = []

    def emit(step: AgentStep) -> AgentStep:
        steps.append(step)
        if on_step is not None:
            on_step(step)
        return step

    agent = CareerPilotAgent(model_name=model_name)
    rank_n = min(max(len(jobs), 1), 20)

    try:
        # --- 1. Recall: pull prior learnings from Phoenix via MCP --------------
        memory_rubric, memory_notes, memory_used = _recall_memory(emit)

        # --- 1b. Model-selected tools: the agent decides what extra context to
        #         gather (Phoenix history and/or local job-pool stats) ----------
        tool_notes = _model_select_tools(agent, profile, jobs, emit)
        plan_context = "\n".join(part for part in (memory_notes, tool_notes) if part)

        # --- 2. Plan ----------------------------------------------------------
        plan = _plan(agent, profile, jobs, plan_context, emit)

        # --- 3. Rank + evaluate (iteration 1) ---------------------------------
        rubric = memory_rubric or default_rubric()
        iterations: list[AgentIteration] = []
        evaluation: EvaluationReport | None = None

        for iteration in range(1, MAX_IMPROVE_ITERS + 2):
            ranking = _rank(agent, profile, jobs, rubric, rank_n, iteration, emit)
            evaluation = _evaluate(agent, profile, jobs, ranking, iteration, emit)
            iterations.append(
                AgentIteration(
                    index=iteration,
                    rubric_version=rubric.version,
                    ranking=ranking,
                    evaluation=evaluation,
                )
            )

            if not needs_improvement(evaluation, EVAL_THRESHOLD):
                stop_reason = (
                    f"Quality threshold met (min score "
                    f"{_score_floor(evaluation)} >= {EVAL_THRESHOLD}) after "
                    f"{iteration} pass(es)."
                )
                break
            if iteration >= MAX_IMPROVE_ITERS + 1:
                stop_reason = (
                    f"Stopped at iteration budget ({MAX_IMPROVE_ITERS} improvement "
                    f"passes); best min score {_score_floor(evaluation)}."
                )
                break

            # --- 4. Self-improve: evaluator-triggered rubric upgrade ----------
            rubric = _improve(agent, rubric, evaluation, profile, jobs, iteration, emit)
        else:  # pragma: no cover - loop always breaks above
            stop_reason = "Stopped."

        emit(
            AgentStep(
                kind="stop",
                title="Agent stopped autonomously",
                detail=stop_reason,
                data={"iterations": len(iterations), "min_score": _score_floor(evaluation)},
            )
        )

        # --- 5. Persist learned rubric back to Phoenix (cross-run memory) -----
        memory_written = _persist_memory(rubric, evaluation, emit)

        final = iterations[-1]
        return AgentRun(
            profile=profile,
            jobs=jobs,
            plan=plan,
            iterations=iterations,
            final_ranking=final.ranking,
            final_evaluation=final.evaluation,
            stop_reason=stop_reason,
            memory_used=memory_used,
            memory_written=memory_written,
            steps=steps,
        )
    finally:
        agent.close()


# --------------------------------------------------------------------------- #
# Step implementations
# --------------------------------------------------------------------------- #


def _recall_memory(emit: StepSink) -> tuple[RankingRubric | None, str, bool]:
    if not mcp_client.is_configured():
        emit(
            AgentStep(
                kind="recall",
                title="Memory: Phoenix not configured",
                detail="Running without cross-run memory (local only).",
                status="warn",
            )
        )
        return None, "", False

    rubric: RankingRubric | None = None
    notes_parts: list[str] = []
    used = False

    try:
        stored = mcp_client.read_rubric_memory()
        if stored:
            rubric = RankingRubric.model_validate(stored)
            base_version = rubric.version.replace("memory:", "").strip() or "baseline"
            rubric.version = f"memory:{base_version}"
            used = True
            notes_parts.append(
                "A previously learned rubric was recalled from Phoenix and used to warm-start ranking."
            )
    except Exception:
        rubric = None

    try:
        spans = mcp_client.recent_eval_spans(limit=15)
        if spans:
            used = True
            gist = "; ".join(
                f"{s.get('step', '')}: {str(s.get('output', ''))[:120]}" for s in spans[:5]
            )
            notes_parts.append(f"Recent run history from Phoenix: {gist}")
    except Exception:
        spans = []

    detail = (
        "Recalled prior rubric and run history from Phoenix MCP."
        if used
        else "No prior memory found in Phoenix yet (first run)."
    )
    emit(
        AgentStep(
            kind="recall",
            title="Recall memory from Phoenix (MCP)",
            detail=detail,
            data={"memory_used": used},
            status="ok" if used else "warn",
        )
    )
    return rubric, "\n".join(notes_parts), used


def _model_select_tools(
    agent: CareerPilotAgent,
    profile: CandidateProfile,
    jobs: list[JobPosting],
    emit: StepSink,
) -> str:
    """Let Gemini choose which tools to call before planning (function calling).

    The model is given two real tools and decides, autonomously, whether to call
    either, both, or none. This is genuine model-selected tool use / dynamic
    routing. Tool results are returned as plain-text notes for the planner. Any
    failure degrades to "no tools used" so the loop never breaks.
    """

    from google.genai import types

    tools = [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="query_run_history",
                    description=(
                        "Query Arize Phoenix for evaluation and feedback history from "
                        "PAST CareerPilot runs. Call this when learnings from prior "
                        "rankings could improve this candidate's strategy."
                    ),
                    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
                ),
                types.FunctionDeclaration(
                    name="inspect_job_pool",
                    description=(
                        "Compute quick statistics about the current job pool (counts of "
                        "visa-friendly, senior, and German-required postings). Call this "
                        "to understand the shape of the pool before ranking."
                    ),
                    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
                ),
            ]
        )
    ]

    prompt = (
        "You are CareerPilot's triage step. Decide which tools (if any) would help "
        "you rank jobs well for this candidate, and call them. You may call zero, "
        "one, or both.\n\n"
        f"Candidate target roles: {', '.join(profile.target_roles) or 'unclear'}\n"
        f"Visa status: {profile.visa_status or 'unclear'}\n"
        f"Jobs in pool: {len(jobs)}\n"
    )

    try:
        response = agent.client.models.generate_content(
            model=agent.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(tools=tools, temperature=0.2),
        )
        calls = _function_calls(response)
    except Exception as exc:  # noqa: BLE001
        emit(
            AgentStep(
                kind="tool",
                title="Tool selection skipped",
                detail=f"Function-calling unavailable: {exc}",
                status="warn",
            )
        )
        return ""

    if not calls:
        emit(
            AgentStep(
                kind="tool",
                title="Model chose no extra tools",
                detail="Gemini decided it had enough context to plan directly.",
            )
        )
        return ""

    notes: list[str] = []
    for name in calls:
        if name == "query_run_history":
            spans = mcp_client.recent_eval_spans(limit=15) if mcp_client.is_configured() else []
            summary = (
                "; ".join(f"{s.get('step', '')}" for s in spans[:6]) or "no prior history"
            )
            notes.append(f"Phoenix run history (model-requested): {summary}")
            emit(
                AgentStep(
                    kind="tool",
                    title="Model called tool: query_run_history (Phoenix MCP)",
                    detail=f"Retrieved {len(spans)} prior run spans.",
                    data={"tool": name, "spans": len(spans)},
                )
            )
        elif name == "inspect_job_pool":
            stats = _inspect_job_pool(jobs)
            notes.append(f"Job pool stats (model-requested): {stats}")
            emit(
                AgentStep(
                    kind="tool",
                    title="Model called tool: inspect_job_pool",
                    detail=stats,
                    data={"tool": name},
                )
            )
    return "\n".join(notes)


def _function_calls(response: Any) -> list[str]:
    names: list[str] = []
    try:
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    names.append(fc.name)
    except Exception:
        return []
    return names


def _inspect_job_pool(jobs: list[JobPosting]) -> str:
    visa = sum(1 for j in jobs if j.visa_blue_card_signals)
    senior = sum(1 for j in jobs if "senior" in (j.seniority or "").lower())
    german = sum(
        1
        for j in jobs
        if any("german" in str(req).lower() for req in j.language_requirements)
    )
    return (
        f"{len(jobs)} jobs: {visa} visa-friendly, {senior} senior-level, "
        f"{german} require German."
    )


def _plan(
    agent: CareerPilotAgent,
    profile: CandidateProfile,
    jobs: list[JobPosting],
    memory_notes: str,
    emit: StepSink,
) -> str:
    prompt = (
        "You are CareerPilot's planning step. In 2-4 short sentences, state the "
        "ranking strategy for THIS candidate: which dimensions matter most "
        "(roles, skills, visa/Blue Card, German level, seniority, location) and "
        "what risks to watch. Be concrete and specific to the candidate.\n\n"
        f"Candidate profile:\n{profile.model_dump_json(indent=2)}\n\n"
        f"Number of jobs to rank: {len(jobs)}\n"
    )
    if memory_notes:
        prompt += f"\nMemory from past runs to incorporate:\n{memory_notes}\n"

    try:
        response = agent.client.models.generate_content(
            model=agent.model_name, contents=prompt
        )
        plan = (response.text or "").strip() or "Rank jobs by overall fit and surface risks."
    except Exception as exc:  # noqa: BLE001
        plan = "Rank jobs by overall fit and surface visa, language, and seniority risks."
        emit(
            AgentStep(
                kind="plan",
                title="Plan (fallback)",
                detail=f"Planner call failed: {exc}",
                status="warn",
            )
        )
        return plan

    emit(AgentStep(kind="plan", title="Plan the ranking strategy", detail=plan))
    return plan


def _rank(
    agent: CareerPilotAgent,
    profile: CandidateProfile,
    jobs: list[JobPosting],
    rubric: RankingRubric,
    rank_n: int,
    iteration: int,
    emit: StepSink,
) -> RankingResult:
    from prompts import ranking_prompt
    from agent import ExtractedJobs

    def _retry_note(attempt: int, err: str) -> None:
        emit(
            AgentStep(
                kind="rank",
                title=f"Self-correcting ranking output (attempt {attempt})",
                detail=err[:200],
                status="warn",
            )
        )

    prompt = ranking_prompt(
        profile_json=profile.model_dump_json(indent=2),
        jobs_json=ExtractedJobs(jobs=jobs).model_dump_json(indent=2),
        rubric=rubric,
        top_n=rank_n,
    )
    ranking = agent.generate_structured_with_retry(
        prompt, RankingResult, attempts=2, on_retry=_retry_note
    )
    ranking.rubric_version = rubric.version
    emit(
        AgentStep(
            kind="rank",
            title=f"Rank jobs (pass {iteration}, rubric '{rubric.version}')",
            detail=f"Produced {len(ranking.matches)} ranked matches.",
            data={"matches": len(ranking.matches), "rubric_version": rubric.version},
        )
    )
    return ranking


def _evaluate(
    agent: CareerPilotAgent,
    profile: CandidateProfile,
    jobs: list[JobPosting],
    ranking: RankingResult,
    iteration: int,
    emit: StepSink,
) -> EvaluationReport:
    report = evaluate_ranking(agent, profile, jobs, ranking)
    emit(
        AgentStep(
            kind="evaluate",
            title=f"Self-evaluate ranking (pass {iteration})",
            detail=(
                f"fit={report.fit_quality} risk={report.risk_detection} "
                f"action={report.actionability} (min {_score_floor(report)}/5)"
            ),
            data={
                "fit_quality": report.fit_quality,
                "risk_detection": report.risk_detection,
                "actionability": report.actionability,
                "weaknesses": report.weaknesses,
            },
            status="ok" if not needs_improvement(report, EVAL_THRESHOLD) else "warn",
        )
    )
    return report


def _improve(
    agent: CareerPilotAgent,
    rubric: RankingRubric,
    evaluation: EvaluationReport,
    profile: CandidateProfile,
    jobs: list[JobPosting],
    iteration: int,
    emit: StepSink,
) -> RankingRubric:
    improved = generate_improved_rubric(agent, rubric, evaluation, profile, jobs)
    if improved.version in (rubric.version, "baseline", "improved"):
        improved.version = f"improved-{iteration}"
    emit(
        AgentStep(
            kind="improve",
            title=f"Improve rubric (triggered by low scores, pass {iteration})",
            detail=(
                "Weaknesses addressed: " + "; ".join(evaluation.weaknesses[:3])
                if evaluation.weaknesses
                else "Rubric tightened for a stronger re-rank."
            ),
            data={"new_rubric_version": improved.version, "weights": improved.weights.model_dump()},
        )
    )
    return improved


def _persist_memory(
    rubric: RankingRubric,
    evaluation: EvaluationReport | None,
    emit: StepSink,
) -> bool:
    if not mcp_client.is_configured():
        return False
    payload = rubric.model_dump()
    payload["learned_at_min_score"] = _score_floor(evaluation)
    note = (
        f"Learned rubric (min eval score {_score_floor(evaluation)}/5)."
        if evaluation
        else "Learned rubric."
    )
    written = mcp_client.write_rubric_memory(json.dumps(payload, indent=2), note=note)
    emit(
        AgentStep(
            kind="memory",
            title="Persist learning to Phoenix (MCP)"
            if written
            else "Persist learning skipped",
            detail=(
                f"Saved rubric '{rubric.version}' as Phoenix prompt "
                f"'{mcp_client.RUBRIC_MEMORY_PROMPT}' for future runs."
            )
            if written
            else "Could not write rubric memory to Phoenix; continuing.",
            status="ok" if written else "warn",
        )
    )
    return written
