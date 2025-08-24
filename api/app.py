"""FastAPI application factory with model lifecycle management."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from api.middleware import register_middleware
from api.routes import router
from logging_config import setup_logging
from training.config import APIConfig, PipelineConfig

logger = logging.getLogger(__name__)

_settings: PipelineConfig | None = None


def get_settings() -> PipelineConfig:
    """Return cached pipeline settings."""
    global _settings
    if _settings is None:
        _settings = PipelineConfig()
    return _settings


def load_whisper_model(
    model_path: Path,
    device: str,
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """
    Load Whisper processor and model from checkpoint.

    Args:
        model_path: Path to saved model directory.
        device: Target device ('cuda' or 'cpu').

    Returns:
        Tuple of (processor, model).

    Raises:
        FileNotFoundError: If model path does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info("Loading Whisper model from %s on %s", model_path, device)
    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    return processor, model


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan: load model on startup, cleanup on shutdown.
    """
    config = get_settings()
    api_cfg: APIConfig = config.api

    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_dir / "api.log",
        structured=config.logging.structured,
    )

    app.state.start_time = time.time()
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.default_language = api_cfg.default_language
    app.state.metrics = {
        "total_requests": 0,
        "transcription_count": 0,
        "error_count": 0,
        "latencies_ms": [],
        "model_name": str(api_cfg.model_path),
    }

    app.state.model_loaded = False
    app.state.processor = None
    app.state.model = None

    model_path = api_cfg.model_path
    if not model_path.is_absolute():
        model_path = config.project_root / model_path

    try:
        processor, model = load_whisper_model(model_path, app.state.device)
        app.state.processor = processor
        app.state.model = model
        app.state.model_loaded = True
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning(
            "Model not found at %s — API will run in degraded mode. "
            "Train a model or set API_MODEL_PATH.",
            model_path,
        )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)

    yield

    app.state.model = None
    app.state.processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Application shutdown complete")


def create_app(config: PipelineConfig | None = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Optional pipeline config override.

    Returns:
        Configured FastAPI app.
    """
    global _settings
    if config is not None:
        _settings = config

    settings = get_settings()

    app = FastAPI(
        title="Whisper ASR API",
        description=(
            "Production REST API for fine-tuned Whisper speech recognition. "
            "Supports low-resource languages including Amharic and Afaan Oromo."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    register_middleware(app, rate_limit=settings.api.rate_limit)
    app.include_router(router)

    # Expose max upload size for routes
    app.state.max_upload_bytes = settings.api.max_upload_size_mb * 1024 * 1024

    return app


app = create_app()
# model.eval() called once in lifespan startup, not per inference
