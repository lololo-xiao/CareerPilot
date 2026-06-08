CareerPilot
===========

CareerPilot is a minimal Streamlit prototype for ranking German job
opportunities for international graduates.

The app:

- extracts a structured profile from an uploaded CV PDF, pasted CV fallback,
  and extra candidate notes
- controls which profile dimensions are extracted through `profile.json`
- separates CV upload, candidate profile review, sourcing, ranking, and
  feedback into distinct Streamlit pages
- stores sourced job links in a session job pool that can be selected for ranking
- can enrich sourced links by fetching job pages and using Gemini to extract
  title, company, location, seniority, and ranking-ready job descriptions
- extracts structured job objects from descriptions separated by `---JOB---`
- ranks jobs with Gemini using a Germany-focused fit and risk rubric
- collects human feedback about visa, language, seniority, location, and realism
- converts feedback into explicit ranking constraints and score caps
- reranks jobs locally and shows a before/after comparison
- records a local trace of the demo steps, with a Phoenix/Arize hook isolated in
  `observability.py`

CareerPilot does not claim that the model is trained or permanently learns. The
MVP demonstrates an observable feedback loop: first match and risk explanation,
human feedback, explicit constraint update, and reranking with a stricter
user-aligned policy.

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

Phoenix / Arize
---------------

The app records local trace events by default. For the Arize hackathon track,
configure Phoenix Cloud tracing:

1. Create a Phoenix API key at `https://app.phoenix.arize.com`.
2. In Phoenix settings, copy your space hostname. It usually looks like
   `https://app.phoenix.arize.com/s/your-space`.
3. Set these in `.env`:

```bash
PHOENIX_API_KEY=px_live_...
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space
PHOENIX_PROJECT_NAME=careerpilot
```

When those values are present, `observability.py` initializes Phoenix tracing
with `phoenix.otel.register(..., auto_instrument=True)` and records custom
CareerPilot spans for:

- profile extraction
- job extraction
- initial ranking
- evaluator result
- human feedback
- constraint update
- improved reranking

Phoenix MCP is configured in `.gemini/settings.json`. Replace
`https://app.phoenix.arize.com/s/your-space` and the empty API key with your
Phoenix values, then restart Gemini CLI from this repo root.

If the app shows stale results after code changes, click **Reset demo** or
restart Streamlit. Old Streamlit session objects may not include newly added
fields.

Files
-----

- `app.py`: Streamlit shell, shared UI helpers, and in-memory workflow helpers
- `pages/`: step-based Streamlit pages for CV upload, candidate profile,
  sourcing, ranking, and feedback
- `agent.py`: Pydantic models, Gemini client setup, extraction, and ranking
- `constraints.py`: human feedback constraints, score caps, and deterministic reranking
- `evaluators.py`: evaluator scoring for first-pass quality signals
- `observability.py`: local trace events and Phoenix/Arize integration hook
- `profile.json`: controlled profile metrics extracted before ranking
- `prompts.py`: prompt templates
- `.env.example`: local configuration template
- `.gemini/settings.json`: Phoenix MCP config template for Gemini CLI
