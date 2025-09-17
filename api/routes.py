"""FastAPI route definitions for ASR inference."""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import librosa
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}
TARGET_SAMPLE_RATE = 16000


class TranscriptionResponse(BaseModel):
    """Response schema for /transcribe."""

    transcription: str
    language: str
    duration_seconds: float = Field(description="Audio duration in seconds")
    inference_time_ms: float = Field(description="Model inference time in milliseconds")


class HealthResponse(BaseModel):
    """Response schema for /health."""

    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response schema for /metrics."""

    total_requests: int
    transcription_count: int
    error_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    model_name: str
    device: str


def _validate_audio_file(filename: str | None) -> None:
    """Validate uploaded file extension."""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Filename is required",
        )
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported audio format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )


async def _load_audio_from_upload(
    file: UploadFile,
    max_size: int = 50 * 1024 * 1024,
) -> tuple[np.ndarray, float]:
    """
    Load and resample audio from uploaded file.

    Returns:
        Tuple of (waveform array, duration in seconds).
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Empty audio file",
        )
    if len(contents) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {max_size // (1024*1024)}MB",
        )

    try:
        audio, sr = librosa.load(io.BytesIO(contents), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to decode audio: {exc}",
        ) from exc

    duration = len(audio) / TARGET_SAMPLE_RATE
    if duration < 0.1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio too short (minimum 0.1 seconds)",
        )
    if duration > 600:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio too long (maximum 600 seconds)",
        )

    return audio, duration


@router.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check(request: Request) -> HealthResponse:
    """Return service health status."""
    app_state = request.app.state
    uptime = time.time() - app_state.start_time
    return HealthResponse(
        status="healthy" if app_state.model_loaded else "degraded",
        model_loaded=app_state.model_loaded,
        device=app_state.device,
        uptime_seconds=round(uptime, 2),
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def get_metrics(request: Request) -> MetricsResponse:
    """Return model and API performance metrics."""
    metrics: dict[str, Any] = request.app.state.metrics
    latencies = metrics.get("latencies_ms", [])

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    return MetricsResponse(
        total_requests=metrics.get("total_requests", 0),
        transcription_count=metrics.get("transcription_count", 0),
        error_count=metrics.get("error_count", 0),
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        model_name=metrics.get("model_name", "unknown"),
        device=request.app.state.device,
    )


@router.post("/transcribe", response_model=TranscriptionResponse, tags=["inference"])
async def transcribe(
    request: Request,
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
) -> TranscriptionResponse:
    """
    Transcribe uploaded audio file using fine-tuned Whisper model.

    Accepts WAV, FLAC, MP3, OGG, M4A, and WebM formats.
    """
    if not request.app.state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is starting up or model path is invalid.",
        )

    _validate_audio_file(audio_file.filename)
    max_upload = getattr(request.app.state, "max_upload_bytes", 50 * 1024 * 1024)
    audio, duration = await _load_audio_from_upload(audio_file, max_size=max_upload)

    processor = request.app.state.processor
    model = request.app.state.model
    device = request.app.state.device
    language = request.app.state.default_language

    start = time.perf_counter()
    try:
        inputs = processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            if hasattr(model, "generate"):
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                )
            else:
                raise RuntimeError("Model does not support generate()")

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0].strip()

    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc

    inference_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Transcribed %.2fs audio in %.1fms",
        duration,
        inference_ms,
    )

    return TranscriptionResponse(
        transcription=transcription,
        language=language,
        duration_seconds=round(duration, 3),
        inference_time_ms=round(inference_ms, 2),
    )
# 503 returned with detail message when app.state.model_loaded is False
# duration validated in _load_audio_from_upload before inference begins
# _validate_audio_file raises 422 when filename is None or empty string
# max_upload_bytes checked before librosa.load to prevent decompression bomb
