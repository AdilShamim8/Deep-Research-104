# ──────────────────────────────────────────────────────────────────────────────
# Deep Research 104 — production Dockerfile
# Multi-stage build: base -> production (slim) + development (with dev tools)
# Target date: 17 October 2026
# ──────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11-slim
ARG BUILDPLATFORM=linux/amd64

# ── Base stage: shared deps + system packages ──────────────────────────────
FROM --platform=${BUILDPLATFORM} python:${PYTHON_VERSION} AS base

# Metadata
LABEL org.opencontainers.image.title="deep-research-104" \
      org.opencontainers.image.authors="Adil Shamim <adilshamim8@users.noreply.github.com>" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.created="2026-10-17T00:00:00Z" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/AdilShamim8/Deep-Research-104"

# ── System dependencies ───────────────────────────────────────────────────────
# curl: healthcheck; git: pip from VCS; build-essential: native ext (lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

# ── Python config ────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HASHCHECK=0

# ── Install Python dependencies (cached layer) ──────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install Playwright Chromium (best-effort; not required for default DuckDuckGo)
RUN playwright install chromium --with-deps 2>/dev/null || true


# ── Production stage ──────────────────────────────────────────────────────────
FROM base AS production

# Copy application code (respect .dockerignore)
COPY . .

# Create non-root user (UID 1000)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Create runtime directories owned by appuser
RUN mkdir -p logs .cache data models && chown -R appuser:appuser logs .cache data models

USER appuser

EXPOSE 8000

# Healthcheck via the /health endpoint we added
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Production command: 4 workers, no reload, info log level
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log", \
     "--no-server-header", \
     "--proxy-headers"]


# ── Development stage ────────────────────────────────────────────────────────
FROM base AS development

# Dev-only deps: tests + linting + debug helpers
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY . .

# Dev runs as root for easier volume mounts
EXPOSE 8000

# Dev command: hot reload, debug logs, single worker
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug", \
     "--proxy-headers"]
