"""Custom log-mel feature extraction with optional delta features."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

logger = logging.getLogger(__name__)

WHISPER_N_MELS = 80
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_SAMPLE_RATE = 16000
WHISPER_CHUNK_LENGTH = 30  # seconds
WHISPER_N_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH


@dataclass
class FeatureConfig:
    """Configuration for log-mel feature extraction."""

    sample_rate: int = WHISPER_SAMPLE_RATE
    n_fft: int = WHISPER_N_FFT
    hop_length: int = WHISPER_HOP_LENGTH
    n_mels: int = WHISPER_N_MELS
    fmin: float = 0.0
    fmax: float = 8000.0
    power: float = 2.0
    normalize: bool = True
    add_deltas: bool = False
    pad_to_max_length: bool = True
    max_length_samples: int = WHISPER_N_SAMPLES


class LogMelExtractor:
    """
    Compute log-mel spectrograms compatible with Whisper preprocessing.

    Optionally appends delta and delta-delta features for improved
    temporal modelling.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """
        Initialize extractor.

        Args:
            config: Feature extraction parameters.
        """
        self.config = config or FeatureConfig()

    def extract(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """
        Compute log-mel spectrogram from waveform.

        Args:
            audio: Mono float32 waveform.
            sr: Sample rate; uses config value if None.

        Returns:
            Log-mel spectrogram, shape (n_mels, time) or (3*n_mels, time) with deltas.
        """
        if sr is not None and sr != self.config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
            sr = self.config.sample_rate

        cfg = self.config

        if cfg.pad_to_max_length:
            audio = self._pad_or_trim(audio, cfg.max_length_samples)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=cfg.power,
        )

        # Convert to log scale (dB), clamp to avoid -inf
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

        if cfg.normalize:
            log_mel = self._normalize(log_mel)

        if cfg.add_deltas:
            delta = librosa.feature.delta(log_mel)
            delta2 = librosa.feature.delta(log_mel, order=2)
            log_mel = np.concatenate([log_mel, delta, delta2], axis=0)

        return log_mel.astype(np.float32)

    def extract_batch(
        self,
        audio_list: list[np.ndarray],
        sr: int | None = None,
    ) -> np.ndarray:
        """
        Extract features for a batch of waveforms.

        Args:
            audio_list: List of mono waveforms.
            sr: Common sample rate.

        Returns:
            Stacked array, shape (batch, features, time).
        """
        features = [self.extract(a, sr) for a in audio_list]
        return np.stack(features, axis=0)

    @staticmethod
    def _pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad with zeros or trim to exact target length."""
        if len(audio) >= target_length:
            return audio[:target_length]
        return np.pad(audio, (0, target_length - len(audio)), mode="constant")

    @staticmethod
    def _normalize(log_mel: np.ndarray) -> np.ndarray:
        """Normalize log-mel to zero mean, unit variance per frame."""
        mean = log_mel.mean(axis=-1, keepdims=True)
        std = log_mel.std(axis=-1, keepdims=True) + 1e-10
        return (log_mel - mean) / std

    def frames_to_seconds(self, n_frames: int) -> float:
        """Convert spectrogram frame count to seconds."""
        return n_frames * self.config.hop_length / self.config.sample_rate

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to spectrogram frame count."""
        return int(seconds * self.config.sample_rate / self.config.hop_length)
