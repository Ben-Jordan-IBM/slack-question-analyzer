FROM python:3.11-slim

WORKDIR /app

# Install the package first so dependency layers cache well
COPY pyproject.toml README.md ./
COPY slack_question_analyzer ./slack_question_analyzer
RUN pip install --no-cache-dir .

# Server and dashboard assets
COPY api_server.py ./
COPY ["Question Analyzer Design System", "./Question Analyzer Design System"]

# The category funnel and the seeded topic bank are DATA the pipeline loads
# at runtime — without these, Docker silently runs a degraded pipeline
# (no routing, no seeded topics) while the README promises both
COPY taxonomy.json seed_topics.json ./

# Bind to all interfaces inside the container; reachable via the mapped port.
# OLLAMA_URL default works with Docker Desktop; compose overrides it.
ENV API_HOST=0.0.0.0 \
    OLLAMA_URL=http://host.docker.internal:11434 \
    NO_BROWSER=1

EXPOSE 5000

CMD ["python", "api_server.py"]
