"""Audio cleaning utilities: silence trimming, normalization, validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Metadata extracted from an audio file."""

    file_path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: str
    rms_energy: float
    peak_amplitude: float
    is_valid: bool
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class CleaningResult:
    """Result of audio cleaning operation."""

    output_path: str
    metadata: AudioMetadata
    trimmed_samples: int
    normalized: bool


class AudioCleaner:
    """
    Clean and validate audio files for ASR training.

    Supports silence trimming, peak normalization, duration validation,
    and invalid audio detection.
    """

    def __init__(
        self,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        top_db: float = 30.0,
        target_db: float = -20.0,
        min_rms: float = 0.001,
    ) -> None:
        """
        Initialize AudioCleaner.

        Args:
            min_duration: Minimum allowed duration in seconds.
            max_duration: Maximum allowed duration in seconds.
            top_db: Threshold in dB below peak for silence trimming.
            target_db: Target RMS level in dB for normalization.
            min_rms: Minimum RMS energy to consider audio non-silent.
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.top_db = top_db
        self.target_db = target_db
        self.min_rms = min_rms

    def load_audio(self, file_path: str | Path) -> tuple[np.ndarray, int]:
        """
        Load audio from disk.

        Args:
            file_path: Path to audio file.

        Returns:
            Tuple of (audio waveform, sample rate).

        Raises:
            ValueError: If file cannot be loaded.
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Audio file not found: {path}")

        try:
            audio, sr = librosa.load(str(path), sr=None, mono=False)
            return audio, sr
        except Exception as exc:
            raise ValueError(f"Failed to load audio {path}: {exc}") from exc

    def extract_metadata(self, file_path: str | Path) -> AudioMetadata:
        """
        Extract metadata and validate audio file.

        Args:
            file_path: Path to audio file.

        Returns:
            AudioMetadata with validation results.
        """
        path = Path(file_path)
        errors: list[str] = []

        try:
            info = sf.info(str(path))
            audio, sr = self.load_audio(path)

            if audio.ndim == 1:
                channels = 1
                waveform = audio
            else:
                channels = audio.shape[0]
                waveform = librosa.to_mono(audio)

            duration = len(waveform) / sr
            rms = float(np.sqrt(np.mean(waveform**2)))
            peak = float(np.max(np.abs(waveform)))

            if duration < self.min_duration:
                errors.append(
                    f"Duration {duration:.2f}s below minimum {self.min_duration}s"
                )
            if duration > self.max_duration:
                errors.append(
                    f"Duration {duration:.2f}s exceeds maximum {self.max_duration}s"
                )
            if rms < self.min_rms:
                errors.append(f"RMS energy {rms:.6f} below threshold {self.min_rms}")
            if peak > 1.0:
                errors.append(f"Clipping detected: peak amplitude {peak:.4f}")

            bit_depth = f"{info.subtype_info}" if info.subtype_info else str(info.subtype)

            return AudioMetadata(
                file_path=str(path),
                duration_seconds=duration,
                sample_rate=sr,
                channels=channels,
                bit_depth=bit_depth,
                rms_energy=rms,
                peak_amplitude=peak,
                is_valid=len(errors) == 0,
                validation_errors=errors,
            )
        except Exception as exc:
            logger.error("Metadata extraction failed for %s: %s", path, exc)
            return AudioMetadata(
                file_path=str(path),
                duration_seconds=0.0,
                sample_rate=0,
                channels=0,
                bit_depth="unknown",
                rms_energy=0.0,
                peak_amplitude=0.0,
                is_valid=False,
                validation_errors=[str(exc)],
            )

    def trim_silence(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        Trim leading and trailing silence.

        Args:
            audio: Audio waveform (mono).
            sr: Sample rate.

        Returns:
            Trimmed audio and number of samples removed.
        """
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        original_len = len(audio)
        trimmed, _ = librosa.effects.trim(audio, top_db=self.top_db)
        removed = original_len - len(trimmed)
        return trimmed, removed

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to target RMS level in dB.

        Args:
            audio: Input waveform.

        Returns:
            Normalized waveform clipped to [-1, 1].
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-10:
            return audio

        target_rms = 10 ** (self.target_db / 20.0)
        normalized = audio * (target_rms / rms)
        return np.clip(normalized, -1.0, 1.0)

    def clean(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        normalize: bool = True,
        trim: bool = True,
    ) -> CleaningResult:
        """
        Clean a single audio file.

        Args:
            input_path: Source audio path.
            output_path: Destination path; defaults to input with _clean suffix.
            normalize: Apply RMS normalization.
            trim: Apply silence trimming.

        Returns:
            CleaningResult with output path and metadata.

        Raises:
            ValueError: If audio is invalid or cannot be processed.
        """
        input_path = Path(input_path)
        metadata = self.extract_metadata(input_path)

        if not metadata.is_valid:
            raise ValueError(
                f"Invalid audio {input_path}: {'; '.join(metadata.validation_errors)}"
            )

        audio, sr = self.load_audio(input_path)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        trimmed_samples = 0
        if trim:
            audio, trimmed_samples = self.trim_silence(audio, sr)

        was_normalized = False
        if normalize:
            audio = self.normalize(audio)
            was_normalized = True

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_clean.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sr, subtype="PCM_16")

        logger.info(
            "Cleaned audio: %s -> %s (trimmed %d samples)",
            input_path,
            output_path,
            trimmed_samples,
        )

        final_metadata = self.extract_metadata(output_path)
        return CleaningResult(
            output_path=str(output_path),
            metadata=final_metadata,
            trimmed_samples=trimmed_samples,
            normalized=was_normalized,
        )

    def clean_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        skip_invalid: bool = True,
    ) -> list[CleaningResult | dict[str, Any]]:
        """
        Clean multiple audio files.

        Args:
            input_paths: List of input file paths.
            output_dir: Directory for cleaned outputs.
            skip_invalid: Skip invalid files instead of raising.

        Returns:
            List of CleaningResult or error dicts for failed files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[CleaningResult | dict[str, Any]] = []

        for input_path in input_paths:
            input_path = Path(input_path)
            out_path = output_dir / f"{input_path.stem}_clean.wav"
            try:
                result = self.clean(input_path, out_path)
                results.append(result)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", input_path, exc)
                if not skip_invalid:
                    raise
                results.append({"file": str(input_path), "error": str(exc)})

        return results

    def is_valid_audio(self, file_path: str | Path) -> bool:
        """Check whether an audio file passes validation."""
        return self.extract_metadata(file_path).is_valid
# stereo -> mono handled via librosa.to_mono in load_audio and clean()
# rms < 1e-10 guard prevents division by zero in normalize()
# load_audio raises ValueError wrapping original error for missing files
