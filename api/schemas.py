"""Pydantic request/response schemas for the ASR API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SupportedLanguage(str, Enum):
    """Languages supported by the fine-tuned Whisper model."""

    AMHARIC = "am"
    OROMO = "om"
    TIGRINYA = "ti"
    ENGLISH = "en"


class TranscriptionRequest(BaseModel):
    """Optional JSON metadata submitted alongside audio uploads."""

    language: SupportedLanguage = SupportedLanguage.AMHARIC
    task: str = Field(default="transcribe", pattern="^(transcribe|translate)$")
    beam_size: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    return_timestamps: bool = False

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v: str) -> str:
        try:
            return SupportedLanguage(v).value
        except ValueError:
            supported = [l.value for l in SupportedLanguage]
            raise ValueError(f"Unsupported language '{v}'. Choose from: {supported}")


class TranscriptionSegment(BaseModel):
    """A timestamped segment from the model output."""

    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Transcribed text for this segment")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TranscriptionResponse(BaseModel):
    """Full response from /transcribe endpoint."""

    transcription: str = Field(description="Complete transcription text")
    language: str
    task: str = "transcribe"
    duration_seconds: float = Field(description="Audio duration in seconds")
    inference_time_ms: float = Field(description="Model inference time in ms")
    segments: list[TranscriptionSegment] | None = None
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model_name: str | None = None


class BatchTranscriptionRequest(BaseModel):
    """Request body for batch /transcribe/batch endpoint."""

    language: SupportedLanguage = SupportedLanguage.AMHARIC
    task: str = "transcribe"
    max_files: int = Field(default=10, ge=1, le=50)


class BatchTranscriptionResponse(BaseModel):
    """Response from /transcribe/batch endpoint."""

    results: list[TranscriptionResponse]
    total_files: int
    successful: int
    failed: int
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    model_name: str | None = None
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Response from /metrics endpoint."""

    total_requests: int
    transcription_count: int
    error_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    model_name: str
    device: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    detail: str
    type: str
    request_id: str | None = None
    docs_url: str | None = None


class APIKeyResponse(BaseModel):
    """Response when creating a new API key."""

    api_key: str = Field(description="Plaintext API key (shown once)")
    label: str
    scopes: list[str]
    warning: str = "Store this key securely. It will not be shown again."
# TranscriptionRequest, BatchTranscriptionRequest moved to dedicated schemas.py
# BatchTranscriptionResponse tracks successful, failed, and total_files counts
# TranscriptionResponse.model_name populated from model.config._name_or_path
