# Built from the repo root (docker compose sets context: .), because eval
# imports the code it measures from the other services (see paths.py). The
# repo layout is kept under /repo so the same relative paths work everywhere.
# Dockerfile.dockerignore next to this file keeps data/ and everything not
# needed out of the build context.
FROM python:3.11-slim

WORKDIR /repo/services/eval
COPY services/eval/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/agent /repo/services/agent
COPY services/vision /repo/services/vision
COPY mcp_servers/pricing /repo/mcp_servers/pricing
COPY scripts /repo/scripts
# The red-team generators draw the face fixture into PII cases.
COPY services/guardrails/fixtures /repo/services/guardrails/fixtures
COPY services/eval /repo/services/eval
ENV REPO_ROOT=/repo

# Run as a non-root user (security best practice). data/ and results/ are
# mounted at run time; the user needs to write generated cases and results.
RUN useradd --create-home appuser && mkdir -p /repo/data && chown appuser /repo/data /repo/services/eval/results
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
