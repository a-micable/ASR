"""Tests for audio augmentation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.augmentation import AudioAugmenter, AugmentationConfig
from tests.conftest import generate_sine_wave, write_test_wav


class TestAudioAugmenter:
    """Tests for AudioAugmenter."""

    def test_time_stretch(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(duration=1.0)
        stretched = aug.time_stretch(audio, rate=0.9)
        # Stretched at 0.9 -> longer
        assert len(stretched) > len(audio)

    def test_time_stretch_speed_up(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(duration=1.0)
        stretched = aug.time_stretch(audio, rate=1.1)
        assert len(stretched) < len(audio)

    def test_add_gaussian_noise(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(amplitude=0.5)
        noisy = aug.add_gaussian_noise(audio, snr_db=20.0)
        assert noisy.shape == audio.shape
        # Noisy should differ from original
        assert not np.allclose(noisy, audio)

    def test_add_noise_silent_passthrough(self) -> None:
        aug = AudioAugmenter()
        audio = np.zeros(16000, dtype=np.float32)
        result = aug.add_gaussian_noise(audio, snr_db=20.0)
        assert result.shape == audio.shape

    def test_volume_gain_positive(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(amplitude=0.1)
        louder = aug.volume_gain(audio, gain_db=6.0)
        assert np.max(np.abs(louder)) > np.max(np.abs(audio))

    def test_volume_gain_clips_to_one(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(amplitude=0.9)
        amplified = aug.volume_gain(audio, gain_db=20.0)
        assert np.max(np.abs(amplified)) <= 1.0

    def test_speed_perturb_no_change(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave()
        result = aug.speed_perturb(audio, sr=16000, rate=1.0)
        assert len(result) == len(audio)

    def test_pitch_shift(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(duration=1.0)
        shifted = aug.pitch_shift(audio, sr=16000, n_steps=2.0)
        assert len(shifted) == len(audio)

    def test_apply_time_mask(self) -> None:
        aug = AudioAugmenter(AugmentationConfig(time_mask_max_frames=10))
        log_mel = np.ones((80, 300), dtype=np.float32)
        masked = aug.apply_time_mask(log_mel)
        # At least some frames should be zeroed
        assert np.any(masked == 0.0)
        assert masked.shape == log_mel.shape

    def test_apply_freq_mask(self) -> None:
        aug = AudioAugmenter(AugmentationConfig(freq_mask_max_bins=10))
        log_mel = np.ones((80, 300), dtype=np.float32)
        masked = aug.apply_freq_mask(log_mel)
        assert np.any(masked == 0.0)
        assert masked.shape == log_mel.shape

    def test_augment_array_random(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(amplitude=0.5)
        result, aug_type, params = aug.augment_array(audio, sr=16000)
        assert isinstance(result, np.ndarray)
        assert aug_type in {"noise", "stretch", "pitch", "speed", "volume"}
        assert "type" in params

    def test_augment_array_specific_type(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave(amplitude=0.5)
        result, aug_type, _ = aug.augment_array(audio, sr=16000, augmentation_type="noise")
        assert aug_type == "noise"

    def test_augment_array_unknown_type(self) -> None:
        aug = AudioAugmenter()
        audio = generate_sine_wave()
        with pytest.raises(ValueError, match="Unknown augmentation"):
            aug.augment_array(audio, sr=16000, augmentation_type="invalid")

    def test_augment_file(self, tmp_path: Path) -> None:
        aug = AudioAugmenter(AugmentationConfig(augment_probability=1.0))
        src = write_test_wav(tmp_path / "source.wav")
        result = aug.augment_file(src, tmp_path / "augmented.wav")
        assert Path(result.output_path).exists()
        assert result.augmentation_type in {"noise", "stretch", "pitch", "speed", "volume"}

    def test_augment_batch(self, tmp_path: Path) -> None:
        aug = AudioAugmenter(AugmentationConfig(augment_probability=1.0, seed=0))
        files = [write_test_wav(tmp_path / f"f{i}.wav") for i in range(3)]
        results = aug.augment_batch(files, tmp_path / "aug", multiplier=2)
        assert len(results) > 0

    def test_augment_batch_empty(self, tmp_path: Path) -> None:
        aug = AudioAugmenter()
        results = aug.augment_batch([], tmp_path / "out")
        assert results == []
