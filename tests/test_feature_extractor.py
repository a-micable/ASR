"""Tests for log-mel feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from preprocessing.feature_extractor import FeatureConfig, LogMelExtractor
from tests.conftest import generate_sine_wave


class TestLogMelExtractor:
    def test_extract_shape(self) -> None:
        extractor = LogMelExtractor(FeatureConfig(pad_to_max_length=False))
        audio = generate_sine_wave(duration=1.0, sample_rate=16000)
        features = extractor.extract(audio)
        assert features.ndim == 2
        assert features.shape[0] == 80  # n_mels

    def test_extract_padded_shape(self) -> None:
        extractor = LogMelExtractor()  # pad to 30s by default
        audio = generate_sine_wave(duration=1.0, sample_rate=16000)
        features = extractor.extract(audio)
        # 30s * 16000 / 160 hop = 3000 frames
        assert features.shape == (80, 3000)

    def test_extract_with_resampling(self) -> None:
        extractor = LogMelExtractor(FeatureConfig(pad_to_max_length=False))
        audio = generate_sine_wave(duration=1.0, sample_rate=44100)
        features = extractor.extract(audio, sr=44100)
        assert features.shape[0] == 80

    def test_normalization_zero_mean(self) -> None:
        extractor = LogMelExtractor(FeatureConfig(normalize=True, pad_to_max_length=False))
        audio = generate_sine_wave(duration=2.0)
        features = extractor.extract(audio)
        # Per-frame normalized — mean across features should be ~0
        assert abs(features.mean()) < 5.0  # loose bound

    def test_no_normalization(self) -> None:
        extractor = LogMelExtractor(FeatureConfig(normalize=False, pad_to_max_length=False))
        audio = generate_sine_wave(duration=1.0)
        features = extractor.extract(audio)
        assert features is not None

    def test_with_deltas(self) -> None:
        extractor = LogMelExtractor(FeatureConfig(add_deltas=True, pad_to_max_length=False))
        audio = generate_sine_wave(duration=1.0)
        features = extractor.extract(audio)
        assert features.shape[0] == 240  # 3 * 80

    def test_extract_batch(self) -> None:
        extractor = LogMelExtractor(FeatureConfig(pad_to_max_length=False))
        audios = [generate_sine_wave(duration=1.0) for _ in range(3)]
        batch = extractor.extract_batch(audios)
        assert batch.shape[0] == 3
        assert batch.shape[1] == 80

    def test_pad_or_trim_pads(self) -> None:
        short = np.zeros(8000, dtype=np.float32)
        padded = LogMelExtractor._pad_or_trim(short, 16000)
        assert len(padded) == 16000
        assert np.all(padded[8000:] == 0.0)

    def test_pad_or_trim_trims(self) -> None:
        long_audio = np.ones(32000, dtype=np.float32)
        trimmed = LogMelExtractor._pad_or_trim(long_audio, 16000)
        assert len(trimmed) == 16000

    def test_frames_to_seconds(self) -> None:
        extractor = LogMelExtractor()
        seconds = extractor.frames_to_seconds(3000)
        assert abs(seconds - 30.0) < 0.1

    def test_seconds_to_frames(self) -> None:
        extractor = LogMelExtractor()
        frames = extractor.seconds_to_frames(30.0)
        assert frames == 3000

    def test_output_dtype_float32(self) -> None:
        extractor = LogMelExtractor(FeatureConfig(pad_to_max_length=False))
        audio = generate_sine_wave()
        features = extractor.extract(audio)
        assert features.dtype == np.float32
