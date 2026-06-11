# CareerPilot — Streamlit agent + Phoenix MCP server in one Cloud Run container.
# Needs BOTH Python (the app) and Node (the @arizeai/phoenix-mcp server the
# runtime spawns over stdio), so we start from a slim Python image and add Node.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Node.js (for `npx @arizeai/phoenix-mcp`) + build essentials.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Pre-install the Phoenix MCP server so the first request doesn't pay a cold
# npm download. `npx @arizeai/phoenix-mcp` then resolves to this global copy.
RUN npm install -g @arizeai/phoenix-mcp@latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

# Cloud Run sets $PORT; Streamlit must bind 0.0.0.0.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false"]
