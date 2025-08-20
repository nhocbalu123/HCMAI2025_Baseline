# Stage 1: Builder
FROM python:3.12.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv with force reinstall for compiled packages
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --no-dev --compile-bytecode

# Stage 2: Runtime
FROM python:3.12.11-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos '' webappnonroot

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/app \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

RUN chown -R webappnonroot:webappnonroot /app

# Switch to non-root user first
USER webappnonroot

# Copy application code (do this after dependencies for better caching)
COPY --chown=webappnonroot:webappnonroot . .

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=10)" || exit 1

EXPOSE 8000

# Use production-ready settings by default, ensure we're in the right directory
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
