"""Audio resampling utilities for Whisper (16 kHz mono)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


@dataclass
class ResamplingConfig:
    """Configuration for audio resampling."""

    target_sample_rate: int = WHISPER_SAMPLE_RATE
    mono: bool = True
    output_format: str = "wav"
    output_subtype: str = "PCM_16"


@dataclass
class ResamplingResult:
    """Result of a resampling operation."""

    input_path: str
    output_path: str
    original_sample_rate: int
    target_sample_rate: int
    duration_seconds: float


class AudioResampler:
    """
    Resample audio to Whisper-compatible format (16 kHz mono).

    Supports single-file and batch processing with optional parallelism.
    """

    def __init__(self, config: ResamplingConfig | None = None) -> None:
        """
        Initialize resampler.

        Args:
            config: Resampling configuration; uses defaults if None.
        """
        self.config = config or ResamplingConfig()

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert multi-channel audio to mono.

        Args:
            audio: Input waveform, shape (channels, samples) or (samples,).

        Returns:
            Mono waveform.
        """
        if audio.ndim == 1:
            return audio
        return librosa.to_mono(audio)

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int | None = None,
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Input waveform.
            orig_sr: Original sample rate.
            target_sr: Target sample rate; defaults to config value.

        Returns:
            Resampled waveform.
        """
        target_sr = target_sr or self.config.target_sample_rate
        if orig_sr == target_sr:
            return audio
        return librosa.resample(
            y=audio,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="scipy",
        )

    def process_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        """
        Convert array to target format (mono + resample).

        Args:
            audio: Input waveform.
            sample_rate: Original sample rate.

        Returns:
            Processed waveform and target sample rate.
        """
        if self.config.mono and audio.ndim > 1:
            audio = self.to_mono(audio)

        audio = self.resample(audio, sample_rate, self.config.target_sample_rate)
        return audio, self.config.target_sample_rate

    def resample_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> ResamplingResult:
        """
        Resample a single audio file to disk.

        Args:
            input_path: Source audio path.
            output_path: Output path; auto-generated if None.

        Returns:
            ResamplingResult with paths and metadata.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        audio, orig_sr = librosa.load(str(input_path), sr=None, mono=False)
        processed, target_sr = self.process_array(audio, orig_sr)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_16k.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(output_path),
            processed,
            target_sr,
            subtype=self.config.output_subtype,
        )

        duration = len(processed) / target_sr
        logger.debug("Resampled %s -> %s (%.2fs)", input_path, output_path, duration)

        return ResamplingResult(
            input_path=str(input_path),
            output_path=str(output_path),
            original_sample_rate=orig_sr,
            target_sample_rate=target_sr,
            duration_seconds=duration,
        )

    def resample_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        max_workers: int = 4,
    ) -> list[ResamplingResult]:
        """
        Resample multiple files, optionally in parallel.

        Args:
            input_paths: List of input file paths.
            output_dir: Output directory.
            max_workers: Thread pool size for parallel processing.

        Returns:
            List of ResamplingResult for each file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def _process(path: str | Path) -> ResamplingResult:
            path = Path(path)
            out = output_dir / f"{path.stem}_16k.{self.config.output_format}"
            return self.resample_file(path, out)

        results: list[ResamplingResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process, p): p for p in input_paths}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    path = futures[future]
                    logger.error("Failed to resample %s: %s", path, exc)
                    raise

        logger.info("Resampled %d files to %s", len(results), output_dir)
        return results
# librosa.load handles M4A via soundfile with ffmpeg fallback for AAC
