CareerPilot
===========

CareerPilot is an **autonomous job-matching agent** for international graduates
seeking work in Germany. Built for the **Arize track** of the Google Cloud Rapid
Agent Hackathon, it goes beyond chat: it plans, ranks, self-evaluates, improves
its own ranking rubric, stops on its own, and **learns across runs** using the
**Arize Phoenix MCP server** as its memory and observability backbone.

Google Cloud AI: ranking, extraction, planning, evaluation, and self-improvement
all run on **Gemini** via the Google Gen AI SDK on **Vertex AI** (`vertexai=True`),
satisfying the Google Cloud AI requirement. The partner "superpower" is the
**Arize Phoenix MCP server**, spawned by the application runtime over stdio.

Agent architecture
------------------

The core loop lives in `orchestrator.py` (`run_agent`). One run:

1. **Recall memory (Phoenix MCP)** — reads the last learned ranking rubric
   (`get-latest-prompt`) and recent run history (`get-spans`) to warm-start.
2. **Model-selected tools** — a Gemini function-calling turn where the model
   autonomously decides whether to call `query_run_history` (Phoenix MCP) and/or
   `inspect_job_pool` before planning.
3. **Plan** — Gemini states the ranking strategy for this specific candidate.
4. **Rank** — scores jobs against the rubric, with **autonomous retry** that
   re-prompts itself on a schema-validation failure.
5. **Self-evaluate** — an LLM-judge scores fit, risk detection, and actionability
   (1-5).
6. **Self-improve loop** — while the min score is below `EVAL_THRESHOLD` and the
   iteration budget remains, the agent rewrites its own rubric
   (`generate_improved_rubric`) and re-ranks.
7. **Autonomous stop** — exits when the quality threshold is met or the budget is
   exhausted, recording the stop reason.
8. **Persist memory (Phoenix MCP)** — writes the winning rubric back as a
   versioned Phoenix prompt (`upsert-prompt`) so the next run starts smarter.

Every step is streamed to an **Agent Console** in the UI and mirrored to Phoenix
as a span, so the agent's reasoning is fully observable. After ranking, the user
can still apply explicit human feedback, which is converted into deterministic
constraints and score caps (`constraints.py`) for a transparent before/after
rerank.

Setup
-----

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Configuration
-------------

The default `.env.example` mirrors the local notebook's Vertex AI setup. Update
the project, location, and model as needed.

For Vertex AI, authenticate locally with Google Application Default Credentials
or set `GOOGLE_APPLICATION_CREDENTIALS`.

For Gemini API key mode, set `GEMINI_API_KEY` and leave
`GOOGLE_GENAI_USE_VERTEXAI=false`.

Controlled Profile Metrics
--------------------------

Edit `profile.json` at the repo root to define the product-default profile
metrics CareerPilot must extract before ranking. End users do not edit or see
raw JSON in the app. They upload a CV or paste fallback text, add notes, and
click **Generate profile** to see a structured profile preview.

Each enabled field becomes a required extraction target and is also returned in
`CandidateProfile.controlled_metrics` for downstream ranking logic.

Useful fields include:

- target roles
- years of experience
- core skills
- language level
- visa / Blue Card needs
- location flexibility
- seniority target
- hard constraints
- soft preferences
- uncertainty fields

Sourcing
--------

The sourcing page first parses job links from pasted text or uploaded `.txt` /
`.csv` files. URL slugs are used only as a fallback title. Click **Extract job
details from links** to fetch each page, convert it to readable text, and use
Gemini to extract structured job details for later ranking. Some job boards may
block automated fetches; those jobs stay editable in the source pool so details
can be added manually.

Arize Phoenix (MCP + tracing)
-----------------------------

Phoenix is used two ways, both at runtime:

1. **MCP server (the partner integration).** `mcp_client.py` spawns
   `@arizeai/phoenix-mcp` over stdio using the Python `mcp` SDK and calls its
   tools — `get-latest-prompt` / `get-spans` to recall memory, and
   `upsert-prompt` to persist the learned rubric. This is what makes the agent
   self-improve across runs. `observability.get_previous_feedback_for_session`
   now returns real Phoenix history instead of an empty list.
2. **OpenTelemetry tracing.** `observability.py` calls
   `phoenix.otel.register(..., auto_instrument=True)` and emits a span for every
   agent step (recall, plan, tool calls, rank, evaluate, self-improve, stop,
   memory write), so the full run is visible in the Phoenix dashboard.

Configure a Phoenix Cloud space:

1. Create a Phoenix API key at `https://app.phoenix.arize.com`.
2. Copy your space hostname, e.g. `https://app.phoenix.arize.com/s/your-space`.
3. Set in `.env`:

```bash
PHOENIX_API_KEY=px_live_...
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space
PHOENIX_PROJECT_NAME=careerpilot
PHOENIX_RUBRIC_PROMPT=careerpilot_rubric   # Phoenix strips hyphens; use underscores
```

The MCP server is also declared in `.gemini/settings.json` for Gemini CLI use,
but the application runtime spawns its own MCP session via `mcp_client.py` and
does not depend on the CLI config. Node.js / `npx` must be available (it is
bundled into the Docker image for deployment).

If the agent runs fine but shows "Phoenix not configured", the key/endpoint are
missing — the loop degrades gracefully and still runs locally.

Deployment (Cloud Run)
----------------------

`Dockerfile` builds a single container with **both Python and Node**, so the
Phoenix MCP server runs as a subprocess on the hosted URL. Deploy with:

```bash
./deploy.sh   # reads PHOENIX_* and GOOGLE_CLOUD_PROJECT from .env
```

It stores the Phoenix API key in **Secret Manager** (mounted via
`--set-secrets`), enables the required APIs, and deploys from source with Cloud
Build. The Cloud Build/compute service account needs the
`roles/cloudbuild.builds.builder`, `roles/storage.objectViewer`,
`roles/artifactregistry.writer`, and `roles/logging.logWriter` roles, and the
runtime service account needs `roles/aiplatform.user` for Vertex Gemini.

Files
-----

- `orchestrator.py`: the agent loop — recall, model-selected tools, plan, rank,
  self-evaluate, self-improve, autonomous stop, persist memory
- `mcp_client.py`: runtime Arize Phoenix MCP client (stdio) — memory read/write
- `app.py`: Streamlit shell, Agent Console (live step stream), UI helpers
- `pages/`: step-based Streamlit pages for CV upload, candidate profile,
  sourcing, ranking (runs the agent), and feedback
- `agent.py`: Pydantic models, Gemini client, extraction, ranking, retrying
  structured generation
- `constraints.py`: human feedback constraints, score caps, deterministic reranking
- `evaluators.py`: LLM-judge scoring and rubric self-improvement
- `observability.py`: Phoenix OpenTelemetry tracing + MCP-backed feedback recall
- `profile.json`: controlled profile metrics extracted before ranking
- `prompts.py`: prompt templates
- `Dockerfile` / `deploy.sh`: Cloud Run deployment (Python + Node + MCP)
- `scripts/`: MCP connectivity checks and an end-to-end agent smoke test
- `.env.example`: configuration template
- `.gemini/settings.json`: Phoenix MCP config for Gemini CLI (optional)
