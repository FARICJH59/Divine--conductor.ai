# ─────────────────────────────────────────────────────────────────
# Divine Conductor AI — Dockerfile
#
# Multi-stage build:
#   builder  — installs Python dependencies into a venv
#   runtime  — lean image that copies the venv and app code
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for any C-extension wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first (layer-cache friendly)
COPY requirements.txt ./

# Create an isolated venv and install all deps into it
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip --quiet \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime image ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Divine Conductor AI" \
      org.opencontainers.image.description="Agentic multimodal film studio" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN addgroup --system studio && adduser --system --ingroup studio studio

WORKDIR /app

# Bring in the pre-built venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/      ./src/
COPY api/      ./api/
COPY config/   ./config/
COPY main.py   ./

# Output directory (can be overridden by a bind-mount or named volume)
RUN mkdir -p /app/output && chown -R studio:studio /app/output

USER studio

# Expose the FastAPI port
EXPOSE 8000

# Health-check so Docker Compose / orchestrators know when the API is ready
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
