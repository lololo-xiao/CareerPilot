#!/usr/bin/env bash
# Deploy CareerPilot (Streamlit agent + Phoenix MCP server) to Google Cloud Run.
# Builds from source via Cloud Build using the repo Dockerfile (Python + Node).
# The Phoenix API key is stored in Secret Manager (never passed on the CLI), and
# mounted into the service with --set-secrets.
#
# Usage:
#   PHOENIX_API_KEY=... PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/<space> ./deploy.sh
# Values default to your local .env when not exported.
set -euo pipefail

# Load .env for defaults. Direct sourcing handles quoted values and comments.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PROJECT="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE="${CLOUD_RUN_SERVICE:-careerpilot}"
MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
PHX_ENDPOINT="${PHOENIX_COLLECTOR_ENDPOINT:?set PHOENIX_COLLECTOR_ENDPOINT}"
PHX_KEY="${PHOENIX_API_KEY:?set PHOENIX_API_KEY}"
PHX_PROJECT="${PHOENIX_PROJECT_NAME:-careerpilot}"
SECRET_NAME="${PHOENIX_SECRET_NAME:-phoenix-api-key}"

echo "Project : $PROJECT"
echo "Region  : $REGION"
echo "Service : $SERVICE"

echo "Enabling required APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  aiplatform.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  --project "$PROJECT"

# --- Store the Phoenix API key in Secret Manager (value via stdin only) -------
if ! gcloud secrets describe "$SECRET_NAME" --project "$PROJECT" >/dev/null 2>&1; then
  echo "Creating secret $SECRET_NAME ..."
  printf '%s' "$PHX_KEY" | gcloud secrets create "$SECRET_NAME" \
    --project "$PROJECT" --replication-policy=automatic --data-file=-
else
  echo "Adding new version to secret $SECRET_NAME ..."
  printf '%s' "$PHX_KEY" | gcloud secrets versions add "$SECRET_NAME" \
    --project "$PROJECT" --data-file=-
fi

# Grant the Cloud Run runtime service account access to the secret.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
  --project "$PROJECT" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role roles/secretmanager.secretAccessor >/dev/null

echo "Deploying from source (Cloud Build will use ./Dockerfile)..."
gcloud run deploy "$SERVICE" \
  --source . \
  --quiet \
  --project "$PROJECT" \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --service-account "$RUNTIME_SA" \
  --set-secrets "PHOENIX_API_KEY=${SECRET_NAME}:latest" \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=${PROJECT},GOOGLE_CLOUD_LOCATION=global,GEMINI_MODEL=${MODEL},PHOENIX_PROJECT_NAME=${PHX_PROJECT},PHOENIX_RUBRIC_PROMPT=${PHOENIX_RUBRIC_PROMPT:-careerpilot_rubric},EVAL_THRESHOLD=${EVAL_THRESHOLD:-4},MAX_IMPROVE_ITERS=${MAX_IMPROVE_ITERS:-2},PHOENIX_MCP_PACKAGE=@arizeai/phoenix-mcp,PHOENIX_COLLECTOR_ENDPOINT=${PHX_ENDPOINT}"

echo "Done. Service URL:"
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)'
