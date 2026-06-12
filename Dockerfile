# =============================================================================
# Whisper Fine-Tuning Pipeline — Production Dockerfile
# Multi-stage | GPU (CUDA 12.1) | Non-root user | Health check
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: System base — CUDA runtime + Python 3.11 + audio libs
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11 (Ubuntu 22.04 ships 3.10 by default; use deadsnakes PPA)
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    # Audio decoding
    ffmpeg \
    libsndfile1 \
    # Health check probe
    curl \
    && rm -rf /var/lib/apt/lists/* \
    # Make python3.11 the default python / python3
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    # Bootstrap pip for Python 3.11 (python3-pip gives pip for 3.10)
    && python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: Python dependencies (own layer so code changes don't bust cache)
# ---------------------------------------------------------------------------
FROM base AS dependencies

# Copy only the pinned production requirements (no pytest/jupyter bloat)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 wheels first
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Now install everything else from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Application code
# ---------------------------------------------------------------------------
FROM dependencies AS application

# Copy source modules (ordered: rarely-changing first for better cache hits)
COPY config/           config/
COPY logging_config.py .
COPY preprocessing/    preprocessing/
COPY training/         training/
COPY evaluation/       evaluation/
COPY monitoring/       monitoring/
COPY api/              api/
COPY scripts/          scripts/

# Create runtime directories, add a non-root user, and fix ownership
RUN mkdir -p data/raw data/processed data/augmented checkpoints logs/tensorboard \
    && groupadd -r whisper \
    && useradd -r -g whisper -d /app -s /sbin/nologin whisper \
    && chown -R whisper:whisper /app

USER whisper

# Runtime environment — all values can be overridden at docker run / k8s
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_MODEL_PATH=checkpoints/best \
    LOG_LEVEL=INFO \
    LOG_STRUCTURED=true \
    WHISPER_MODEL_LANGUAGE=am \
    WHISPER_MODEL_NAME=openai/whisper-small

EXPOSE 8000

# Health check — uses /bin/sh -c so ${API_PORT} expands correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f "http://localhost:${API_PORT}/health" || exit 1

# Uvicorn with 1 worker by default; override API_WORKERS env for multi-worker
CMD ["sh", "-c", \
     "python -m uvicorn api.app:app \
      --host ${API_HOST} \
      --port ${API_PORT} \
      --workers ${API_WORKERS:-1} \
      --log-level ${LOG_LEVEL:-info}"]
