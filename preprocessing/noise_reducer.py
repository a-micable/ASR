"""Background noise reduction for speech audio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class NoiseReductionConfig:
    """Configuration for noise reduction."""

    stationary: bool = True
    prop_decrease: float = 0.75
    n_fft: int = 2048
    hop_length: int = 512
    n_std_thresh_stationary: float = 1.5
    noise_threshold_db: float = -40.0
    min_signal_rms: float = 0.005


@dataclass
class NoiseReductionResult:
    """Result of noise reduction."""

    input_path: str
    output_path: str
    snr_improvement_db: float
    applied: bool


class NoiseReducer:
    """
    Reduce background noise and enhance signal quality.

    Uses spectral gating via noisereduce with configurable thresholds.
    """

    def __init__(self, config: NoiseReductionConfig | None = None) -> None:
        """
        Initialize noise reducer.

        Args:
            config: Noise reduction settings.
        """
        self.config = config or NoiseReductionConfig()

    def _compute_snr(self, signal: np.ndarray, noise_floor: float = 1e-10) -> float:
        """Estimate SNR in dB from RMS energy."""
        rms = np.sqrt(np.mean(signal**2)) + noise_floor
        return float(20 * np.log10(rms / noise_floor))

    def _estimate_noise_profile(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Estimate noise from the first 0.5 seconds (assumed silence/noise).

        Args:
            audio: Full waveform.
            sr: Sample rate.

        Returns:
            Noise sample segment.
        """
        noise_len = min(int(0.5 * sr), len(audio) // 4)
        if noise_len < 100:
            return audio[: max(len(audio) // 10, 1)]
        return audio[:noise_len]

    def reduce_noise_array(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, bool]:
        """
        Apply noise reduction to a numpy array.

        Args:
            audio: Input waveform (mono).
            sr: Sample rate.

        Returns:
            Denoised audio and whether reduction was applied.
        """
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < self.config.min_signal_rms:
            logger.debug("Signal too weak for noise reduction (rms=%.6f)", rms)
            return audio, False

        snr_before = self._compute_snr(audio)
        if snr_before > abs(self.config.noise_threshold_db):
            logger.debug("SNR sufficient (%.1f dB), skipping reduction", snr_before)
            return audio, False

        noise_clip = self._estimate_noise_profile(audio, sr)

        if self.config.stationary:
            reduced = nr.reduce_noise(
                y=audio,
                y_noise=noise_clip,
                sr=sr,
                stationary=True,
                prop_decrease=self.config.prop_decrease,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_std_thresh_stationary=self.config.n_std_thresh_stationary,
            )
        else:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,
                prop_decrease=self.config.prop_decrease,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )

        return reduced.astype(np.float32), True

    def enhance_signal(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply light pre-emphasis to improve high-frequency clarity.

        Args:
            audio: Input waveform.

        Returns:
            Enhanced waveform.
        """
        coef = 0.97
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        enhance: bool = True,
    ) -> NoiseReductionResult:
        """
        Reduce noise in an audio file and save result.

        Args:
            input_path: Source audio path.
            output_path: Destination path.
            enhance: Apply signal enhancement after denoising.

        Returns:
            NoiseReductionResult with SNR improvement estimate.
        """
        input_path = Path(input_path)
        audio, sr = librosa.load(str(input_path), sr=None, mono=True)

        snr_before = self._compute_snr(audio)
        reduced, applied = self.reduce_noise_array(audio, sr)

        if enhance and applied:
            reduced = self.enhance_signal(reduced)

        snr_after = self._compute_snr(reduced)
        snr_improvement = snr_after - snr_before

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_denoised.wav"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), reduced, sr, subtype="PCM_16")

        logger.info(
            "Noise reduction %s: applied=%s, SNR Δ=%.1f dB",
            input_path.name,
            applied,
            snr_improvement,
        )

        return NoiseReductionResult(
            input_path=str(input_path),
            output_path=str(output_path),
            snr_improvement_db=snr_improvement,
            applied=applied,
        )

    def process_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
    ) -> list[NoiseReductionResult]:
        """
        Process multiple files.

        Args:
            input_paths: Input file paths.
            output_dir: Output directory.

        Returns:
            List of results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for path in input_paths:
            path = Path(path)
            out = output_dir / f"{path.stem}_denoised.wav"
            results.append(self.process_file(path, out))

        return results
