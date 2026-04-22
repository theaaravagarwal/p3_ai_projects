# Multi-stage Dockerfile for the lean deployment image.
# This path is intentionally separate from the local Langflow-backed setup.

FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-docker.txt ./

RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements-docker.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv

COPY src/ src/
COPY README.md DEPLOYMENT.md .env.example main.py ./

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/ >/dev/null || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
