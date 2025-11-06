"""Whisper inference engine with beam search and timestamp support."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from api.schemas import TranscriptionSegment

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
WHISPER_CHUNK_SECONDS = 30
ALLOWED_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"})


@dataclass
class InferenceConfig:
    """Runtime configuration for the inference engine."""

    language: str = "am"
    task: str = "transcribe"
    beam_size: int = 5
    temperature: float = 0.0
    return_timestamps: bool = False
    chunk_length_s: int = WHISPER_CHUNK_SECONDS
    max_new_tokens: int = 448


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    transcription: str
    language: str
    duration_seconds: float
    inference_time_ms: float
    segments: list[TranscriptionSegment] = field(default_factory=list)
    confidence_score: float | None = None
    model_name: str = ""


class WhisperInferenceEngine:
    """
    Stateful Whisper inference engine wrapping processor and model.

    Handles audio preprocessing, chunked long-form transcription,
    and optional timestamp extraction.
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        model: WhisperForConditionalGeneration,
        device: str,
        config: InferenceConfig | None = None,
    ) -> None:
        self.processor = processor
        self.model = model
        self.device = device
        self.config = config or InferenceConfig()
        self.model.eval()

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_audio_bytes(
        data: bytes,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> tuple[np.ndarray, float]:
        """
        Decode audio bytes to a 16 kHz mono float32 array.

        Args:
            data: Raw audio bytes (any format supported by librosa).
            max_bytes: Maximum permitted byte size.

        Returns:
            (waveform, duration_seconds)

        Raises:
            ValueError: For empty, oversized, or corrupt audio.
        """
        if not data:
            raise ValueError("Empty audio data")
        if len(data) > max_bytes:
            raise ValueError(
                f"File size {len(data)} exceeds limit {max_bytes // (1024*1024)} MB"
            )
        try:
            audio, _ = librosa.load(io.BytesIO(data), sr=TARGET_SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise ValueError(f"Failed to decode audio: {exc}") from exc

        duration = len(audio) / TARGET_SAMPLE_RATE
        if duration < 0.1:
            raise ValueError("Audio too short (minimum 0.1 s)")
        if duration > 600:
            raise ValueError("Audio too long (maximum 600 s)")
        return audio, duration

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        duration: float,
        config: InferenceConfig | None = None,
    ) -> InferenceResult:
        """
        Run Whisper transcription on a prepared waveform.

        Args:
            audio: 16 kHz mono float32 waveform.
            duration: Audio duration in seconds.
            config: Override inference config for this call.

        Returns:
            InferenceResult with transcription and metadata.
        """
        cfg = config or self.config
        start = time.perf_counter()

        try:
            inputs = self.processor(
                audio,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.device)

            generate_kwargs: dict[str, Any] = {
                "num_beams": cfg.beam_size,
                "max_new_tokens": cfg.max_new_tokens,
            }

            if cfg.return_timestamps:
                generate_kwargs["return_timestamps"] = True

            if cfg.temperature > 0:
                generate_kwargs["temperature"] = cfg.temperature
                generate_kwargs["do_sample"] = True
            else:
                generate_kwargs["temperature"] = None
                generate_kwargs["do_sample"] = False

            try:
                forced_ids = self.processor.get_decoder_prompt_ids(
                    language=cfg.language, task=cfg.task
                )
                if forced_ids:
                    generate_kwargs["forced_decoder_ids"] = forced_ids
            except Exception:
                pass  # Older processor versions may not support this

            with torch.no_grad():
                outputs = self.model.generate(input_features, **generate_kwargs)

            transcription = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0].strip()

            segments = self._extract_segments(outputs, audio, cfg)

        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        inference_ms = (time.perf_counter() - start) * 1000
        logger.info("Transcribed %.2fs in %.1fms", duration, inference_ms)

        return InferenceResult(
            transcription=transcription,
            language=cfg.language,
            duration_seconds=round(duration, 3),
            inference_time_ms=round(inference_ms, 2),
            segments=segments,
            model_name=getattr(self.model.config, "_name_or_path", "unknown"),
        )

    def _extract_segments(
        self,
        output_ids: torch.Tensor,
        audio: np.ndarray,
        cfg: InferenceConfig,
    ) -> list[TranscriptionSegment]:
        """
        Extract timestamped segments from model output if available.

        Args:
            output_ids: Generated token IDs.
            audio: Original waveform.
            cfg: Inference config.

        Returns:
            List of TranscriptionSegment (may be empty if no timestamps).
        """
        if not cfg.return_timestamps:
            return []

        try:
            # Try to use the processor's built-in timestamp decoding
            decoded = self.processor.decode(output_ids[0], output_offsets=True)
            chunks = decoded.get("chunks", [])
            segments = []
            for chunk in chunks:
                ts = chunk.get("timestamp", (None, None))
                text = chunk.get("text", "").strip()
                if text and ts[0] is not None:
                    segments.append(
                        TranscriptionSegment(
                            start=float(ts[0]),
                            end=float(ts[1]) if ts[1] is not None else float(ts[0]) + 1.0,
                            text=text,
                        )
                    )
            return segments
        except Exception:
            return []

    def transcribe_long(
        self,
        audio: np.ndarray,
        duration: float,
        config: InferenceConfig | None = None,
    ) -> InferenceResult:
        """
        Transcribe audio longer than 30 s using overlapping chunking.

        Args:
            audio: 16 kHz mono waveform (any length).
            duration: Duration in seconds.
            config: Inference configuration.

        Returns:
            Combined InferenceResult.
        """
        cfg = config or self.config
        chunk_samples = cfg.chunk_length_s * TARGET_SAMPLE_RATE
        overlap_samples = 2 * TARGET_SAMPLE_RATE  # 2s overlap

        if len(audio) <= chunk_samples:
            return self.transcribe(audio, duration, cfg)

        parts: list[str] = []
        all_segments: list[TranscriptionSegment] = []
        offset = 0.0
        total_inference_ms = 0.0

        i = 0
        while i < len(audio):
            chunk = audio[i : i + chunk_samples]
            chunk_dur = len(chunk) / TARGET_SAMPLE_RATE
            result = self.transcribe(chunk, chunk_dur, cfg)
            parts.append(result.transcription)
            total_inference_ms += result.inference_time_ms

            for seg in result.segments:
                all_segments.append(
                    TranscriptionSegment(
                        start=seg.start + offset,
                        end=seg.end + offset,
                        text=seg.text,
                    )
                )

            offset += chunk_dur - (overlap_samples / TARGET_SAMPLE_RATE)
            i += chunk_samples - overlap_samples

        return InferenceResult(
            transcription=" ".join(parts).strip(),
            language=cfg.language,
            duration_seconds=round(duration, 3),
            inference_time_ms=round(total_inference_ms, 2),
            segments=all_segments,
        )
# forced_decoder_ids retrieved once per request via processor.get_decoder_prompt_ids
# librosa.load called with sr=TARGET_SAMPLE_RATE to do resampling in one pass
# WhisperInferenceEngine encapsulates processor, model, device and config
# forced_ids=None skipped; generate() called without forced_decoder_ids kwarg
# decoded.get('chunks', []) returns empty list when no chunks key present
# ValueError with 'too long' message triggers 422 via ErrorHandlingMiddleware
