"""Audio data augmentation for low-resource ASR training."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from preprocessing.io_utils import auto_output_path, ensure_output_dir, load_audio, save_audio

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation pipeline."""

    time_stretch_rates: list[float] = field(default_factory=lambda: [0.9, 1.1])
    pitch_shift_steps: list[float] = field(default_factory=lambda: [-2.0, 2.0])
    noise_snr_db_range: tuple[float, float] = (15.0, 30.0)
    speed_perturb_rates: list[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])
    volume_gain_db_range: tuple[float, float] = (-6.0, 6.0)
    time_mask_max_frames: int = 30
    freq_mask_max_bins: int = 20
    augment_probability: float = 0.5
    seed: int = 42


@dataclass
class AugmentationResult:
    """Result from augmenting a single audio sample."""

    input_path: str
    output_path: str
    augmentation_type: str
    augmentation_params: dict


class AudioAugmenter:
    """
    Apply stochastic audio augmentations for ASR data augmentation.

    Supports: time stretching, pitch shifting, additive noise, speed
    perturbation, volume gain, and SpecAugment-style masking on log-mel
    features.
    """

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        self._rng = random.Random(self.config.seed)
        self._np_rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------
    # Waveform-level augmentations
    # ------------------------------------------------------------------

    def time_stretch(self, audio: np.ndarray, rate: float | None = None) -> np.ndarray:
        """Stretch or compress audio in time without changing pitch."""
        if rate is None:
            rate = self._rng.choice(self.config.time_stretch_rates)
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(
        self, audio: np.ndarray, sr: int, n_steps: float | None = None
    ) -> np.ndarray:
        """Shift pitch by semitones without changing duration."""
        if n_steps is None:
            n_steps = float(self._rng.choice(self.config.pitch_shift_steps))
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def add_gaussian_noise(
        self, audio: np.ndarray, snr_db: float | None = None
    ) -> np.ndarray:
        """Add white Gaussian noise at a target SNR level."""
        if snr_db is None:
            lo, hi = self.config.noise_snr_db_range
            snr_db = float(self._np_rng.uniform(lo, hi))
        signal_power = np.mean(audio**2)
        if signal_power < 1e-10:
            return audio
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self._np_rng.normal(0, np.sqrt(noise_power), len(audio))
        return (audio + noise).astype(np.float32)

    def speed_perturb(self, audio: np.ndarray, sr: int, rate: float | None = None) -> np.ndarray:
        """Change speed (and pitch) by resampling."""
        if rate is None:
            rate = self._rng.choice(self.config.speed_perturb_rates)
        if abs(rate - 1.0) < 1e-5:
            return audio
        new_sr = int(sr * rate)
        return librosa.resample(audio, orig_sr=new_sr, target_sr=sr)

    def volume_gain(self, audio: np.ndarray, gain_db: float | None = None) -> np.ndarray:
        """Apply random volume gain in dB."""
        if gain_db is None:
            lo, hi = self.config.volume_gain_db_range
            gain_db = float(self._np_rng.uniform(lo, hi))
        factor = 10 ** (gain_db / 20.0)
        return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Spectrogram-level augmentations (SpecAugment)
    # ------------------------------------------------------------------

    def apply_time_mask(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment time masking to log-mel spectrogram.

        Args:
            log_mel: Shape (n_mels, time_frames).

        Returns:
            Masked spectrogram.
        """
        _, T = log_mel.shape
        t = self._rng.randint(0, min(self.config.time_mask_max_frames, T))
        t0 = self._rng.randint(0, max(T - t, 1))
        masked = log_mel.copy()
        masked[:, t0 : t0 + t] = 0.0
        return masked

    def apply_freq_mask(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment frequency masking to log-mel spectrogram.

        Args:
            log_mel: Shape (n_mels, time_frames).

        Returns:
            Masked spectrogram.
        """
        n_mels, _ = log_mel.shape
        f = self._rng.randint(0, min(self.config.freq_mask_max_bins, n_mels))
        f0 = self._rng.randint(0, max(n_mels - f, 1))
        masked = log_mel.copy()
        masked[f0 : f0 + f, :] = 0.0
        return masked

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def augment_array(
        self,
        audio: np.ndarray,
        sr: int,
        augmentation_type: str = "random",
    ) -> tuple[np.ndarray, str, dict]:
        """
        Apply a single augmentation to a waveform.

        Args:
            audio: Mono float32 waveform.
            sr: Sample rate.
            augmentation_type: One of 'noise', 'stretch', 'pitch', 'speed',
                'volume', or 'random'.

        Returns:
            Augmented waveform, applied type, and parameters used.
        """
        options: list[tuple[str, Callable]] = [
            ("noise", lambda a: self.add_gaussian_noise(a)),
            ("stretch", lambda a: self.time_stretch(a)),
            ("pitch", lambda a: self.pitch_shift(a, sr)),
            ("speed", lambda a: self.speed_perturb(a, sr)),
            ("volume", lambda a: self.volume_gain(a)),
        ]

        if augmentation_type == "random":
            aug_type, fn = self._rng.choice(options)
        else:
            fn_map = dict(options)
            if augmentation_type not in fn_map:
                raise ValueError(f"Unknown augmentation: {augmentation_type}")
            aug_type = augmentation_type
            fn = fn_map[aug_type]

        augmented = fn(audio)
        params: dict = {"type": aug_type}
        return augmented, aug_type, params

    def augment_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        augmentation_type: str = "random",
    ) -> AugmentationResult:
        """
        Augment a single audio file and save result.

        Args:
            input_path: Source audio file.
            output_path: Destination; auto-named if None.
            augmentation_type: Augmentation to apply.

        Returns:
            AugmentationResult.
        """
        input_path = Path(input_path)
        audio, sr = load_audio(input_path, mono=True)
        augmented, aug_type, params = self.augment_array(audio, sr, augmentation_type)

        if output_path is None:
            output_path = auto_output_path(input_path, f"_{aug_type}")
        else:
            output_path = Path(output_path)

        save_audio(augmented, sr, output_path)
        logger.debug("Augmented %s -> %s [%s]", input_path.name, output_path.name, aug_type)

        return AugmentationResult(
            input_path=str(input_path),
            output_path=str(output_path),
            augmentation_type=aug_type,
            augmentation_params=params,
        )

    def augment_batch(
        self,
        input_paths: list[str | Path],
        output_dir: str | Path,
        multiplier: int = 1,
    ) -> list[AugmentationResult]:
        """
        Augment multiple files, optionally generating multiple variants.

        Args:
            input_paths: Source audio files.
            output_dir: Output directory.
            multiplier: Number of augmented copies per file.

        Returns:
            List of AugmentationResult.
        """
        output_dir = ensure_output_dir(output_dir)
        results = []

        for path in input_paths:
            path = Path(path)
            for i in range(multiplier):
                if self._rng.random() > self.config.augment_probability:
                    continue
                out = auto_output_path(path, f"_aug{i}", output_dir)
                try:
                    results.append(self.augment_file(path, out))
                except Exception as exc:
                    logger.warning("Failed to augment %s: %s", path, exc)

        logger.info(
            "Augmented %d files -> %d variants in %s",
            len(input_paths),
            len(results),
            output_dir,
        )
        return results
# all public methods document Args, Returns, and Raises in Google format
# augmentation_type options: 'noise', 'stretch', 'pitch', 'speed', 'volume', 'random'
# librosa.resample returns float64; cast to float32 for memory efficiency
# logger.info('Augmented %d -> %d variants', len(inputs), len(results))
# AugmentationResult(input_path, output_path, augmentation_type, params)
# PitchShift uses librosa.effects.pitch_shift with n_steps in [-2, +2]
