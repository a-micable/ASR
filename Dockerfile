# =============================================================================
# Whisper Fine-Tuning Pipeline — Production Dockerfile
# GPU-compatible | Non-root user | Health checks | Layer caching
# =============================================================================

# --- Stage 1: Base with CUDA runtime ---
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# --- Stage 2: Dependencies (cached layer) ---
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# --- Stage 3: Application ---
FROM dependencies AS application

COPY preprocessing/ preprocessing/
COPY training/ training/
COPY evaluation/ evaluation/
COPY api/ api/
COPY config/ config/
COPY logging_config.py .
COPY scripts/ scripts/

RUN mkdir -p data/raw data/processed data/augmented checkpoints logs && \
    groupadd -r whisper && useradd -r -g whisper -d /app whisper && \
    chown -R whisper:whisper /app

USER whisper

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_MODEL_PATH=checkpoints/best \
    LOG_LEVEL=INFO \
    WHISPER_MODEL_LANGUAGE=am

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
# RUN groupadd -r whisper && useradd -r whisper ensures non-root execution
# three stages: base, dependencies, application for layer cache efficiency
