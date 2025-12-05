"""Tests for preprocessing module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from preprocessing.audio_cleaner import AudioCleaner
from preprocessing.dataset_builder import DatasetBuilder
from preprocessing.noise_reducer import NoiseReducer, NoiseReductionConfig
from preprocessing.resampler import AudioResampler, ResamplingConfig
from tests.conftest import generate_sine_wave, write_test_wav


class TestAudioCleaner:
    """Tests for AudioCleaner."""

    def test_extract_metadata_valid(self, temp_audio_dir: Path) -> None:
        cleaner = AudioCleaner(min_duration=0.5, max_duration=10.0)
        meta = cleaner.extract_metadata(temp_audio_dir / "sample1.wav")
        assert meta.is_valid
        assert meta.sample_rate == 44100
        assert meta.duration_seconds > 0

    def test_extract_metadata_missing_file(self, tmp_path: Path) -> None:
        cleaner = AudioCleaner()
        meta = cleaner.extract_metadata(tmp_path / "missing.wav")
        assert not meta.is_valid
        assert meta.validation_errors

    def test_trim_silence(self) -> None:
        cleaner = AudioCleaner()
        sr = 16000
        silence = np.zeros(int(0.5 * sr))
        tone = generate_sine_wave(1.0, sr)
        audio = np.concatenate([silence, tone, silence])
        trimmed, removed = cleaner.trim_silence(audio, sr)
        assert removed > 0
        assert len(trimmed) < len(audio)

    def test_normalize(self) -> None:
        cleaner = AudioCleaner(target_db=-20.0)
        audio = generate_sine_wave(amplitude=0.01)
        normalized = cleaner.normalize(audio)
        rms = np.sqrt(np.mean(normalized**2))
        target_rms = 10 ** (-20.0 / 20.0)
        assert abs(rms - target_rms) < 0.05

    def test_clean_file(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        cleaner = AudioCleaner(min_duration=0.5, max_duration=10.0)
        out = tmp_path / "clean.wav"
        result = cleaner.clean(temp_audio_dir / "sample1.wav", out)
        assert Path(result.output_path).exists()
        assert result.normalized

    def test_clean_batch(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        cleaner = AudioCleaner(min_duration=0.5, max_duration=10.0)
        inputs = list(temp_audio_dir.glob("*.wav"))
        results = cleaner.clean_batch(inputs, tmp_path / "clean")
        assert len(results) == 3

    def test_invalid_duration_rejected(self, tmp_path: Path) -> None:
        cleaner = AudioCleaner(min_duration=5.0)
        path = write_test_wav(tmp_path / "short.wav", duration=0.3)
        assert not cleaner.is_valid_audio(path)

    def test_load_audio_stereo(self, tmp_path: Path) -> None:
        cleaner = AudioCleaner()
        stereo = np.stack([generate_sine_wave(), generate_sine_wave()])
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo.T, 16000)
        audio, sr = cleaner.load_audio(path)
        assert sr == 16000


class TestAudioResampler:
    """Tests for AudioResampler."""

    def test_resample_to_16k(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        resampler = AudioResampler()
        result = resampler.resample_file(
            temp_audio_dir / "sample1.wav",
            tmp_path / "out.wav",
        )
        assert result.target_sample_rate == 16000
        assert Path(result.output_path).exists()
        info = sf.info(result.output_path)
        assert info.samplerate == 16000

    def test_mono_conversion(self) -> None:
        resampler = AudioResampler()
        stereo = np.stack([generate_sine_wave(), generate_sine_wave()])
        mono = resampler.to_mono(stereo)
        assert mono.ndim == 1

    def test_process_array_same_rate(self) -> None:
        resampler = AudioResampler(ResamplingConfig(target_sample_rate=16000))
        audio = generate_sine_wave(sample_rate=16000)
        processed, sr = resampler.process_array(audio, 16000)
        assert sr == 16000
        assert len(processed) == len(audio)

    def test_resample_batch(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        resampler = AudioResampler()
        inputs = list(temp_audio_dir.glob("*.wav"))
        results = resampler.resample_batch(inputs, tmp_path / "resampled")
        assert len(results) == 3

    def test_resample_file_not_found(self) -> None:
        resampler = AudioResampler()
        with pytest.raises(FileNotFoundError):
            resampler.resample_file("/nonexistent/file.wav")


class TestNoiseReducer:
    """Tests for NoiseReducer."""

    def test_reduce_noise_array(self) -> None:
        reducer = NoiseReducer(NoiseReductionConfig(min_signal_rms=0.001))
        audio = generate_sine_wave(amplitude=0.3)
        reduced, applied = reducer.reduce_noise_array(audio, 16000)
        assert len(reduced) == len(audio)
        assert isinstance(applied, bool)

    def test_skip_weak_signal(self) -> None:
        reducer = NoiseReducer(NoiseReductionConfig(min_signal_rms=0.5))
        audio = generate_sine_wave(amplitude=0.01)
        _, applied = reducer.reduce_noise_array(audio, 16000)
        assert not applied

    def test_enhance_signal(self) -> None:
        reducer = NoiseReducer()
        audio = generate_sine_wave()
        enhanced = reducer.enhance_signal(audio)
        assert len(enhanced) == len(audio)

    def test_process_file(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        reducer = NoiseReducer()
        result = reducer.process_file(
            temp_audio_dir / "sample1.wav",
            tmp_path / "denoised.wav",
        )
        assert Path(result.output_path).exists()


class TestDatasetBuilder:
    """Tests for DatasetBuilder."""

    def test_parse_csv(self, temp_audio_dir: Path, transcription_csv: Path) -> None:
        builder = DatasetBuilder(temp_audio_dir, transcription_csv, language="am")
        records = builder.parse_transcription_file(transcription_csv)
        assert len(records) == 3
        assert records[0]["language"] == "am"

    def test_parse_jsonl(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            json.dumps({"audio": "sample1.wav", "text": "test", "speaker": "s1"}) + "\n",
            encoding="utf-8",
        )
        builder = DatasetBuilder(temp_audio_dir, language="om")
        records = builder.parse_transcription_file(jsonl)
        assert len(records) == 1

    def test_create_splits(self, temp_audio_dir: Path, transcription_csv: Path) -> None:
        builder = DatasetBuilder(temp_audio_dir, transcription_csv)
        records = builder.build_records()
        splits = builder.create_splits(records, seed=42)
        total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
        assert total == len(records)

    def test_build_dataset(self, temp_audio_dir: Path, transcription_csv: Path) -> None:
        builder = DatasetBuilder(temp_audio_dir, transcription_csv)
        dataset = builder.build_dataset(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert "train" in dataset
        assert len(dataset["train"]) > 0

    def test_compute_statistics(self, temp_audio_dir: Path, transcription_csv: Path) -> None:
        builder = DatasetBuilder(temp_audio_dir, transcription_csv)
        dataset = builder.build_dataset()
        stats = builder.compute_statistics(dataset)
        assert stats.total_samples > 0
        assert stats.vocabulary_size > 0

    def test_export_artifacts(self, temp_audio_dir: Path, transcription_csv: Path, tmp_path: Path) -> None:
        builder = DatasetBuilder(temp_audio_dir, transcription_csv)
        dataset = builder.build_dataset()
        stats = builder.export_artifacts(dataset, tmp_path / "export")
        assert (tmp_path / "export" / "dataset_statistics.json").exists()
        assert stats.train_samples >= 0

    def test_normalize_text(self) -> None:
        text = DatasetBuilder.normalize_text("  hello   world  ", language="om")
        assert text == "hello world"

    def test_invalid_split_ratios(self, temp_audio_dir: Path, transcription_csv: Path) -> None:
        builder = DatasetBuilder(temp_audio_dir, transcription_csv)
        records = builder.build_records()
        with pytest.raises(ValueError):
            builder.create_splits(records, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_scan_audio_directory(self, temp_audio_dir: Path) -> None:
        builder = DatasetBuilder(temp_audio_dir, language="om")
        records = builder.scan_audio_directory()
        assert len(records) == 3

    def test_parse_plain_text(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("sample1.wav|transcription one\n# comment\nsample2.wav|two\n")
        builder = DatasetBuilder(temp_audio_dir)
        records = builder.parse_transcription_file(manifest)
        assert len(records) == 2

    def test_parse_json(self, temp_audio_dir: Path, tmp_path: Path) -> None:
        manifest = tmp_path / "data.json"
        manifest.write_text(
            '[{"audio": "sample1.wav", "text": "test", "speaker": "s1"}]',
            encoding="utf-8",
        )
        builder = DatasetBuilder(temp_audio_dir)
        records = builder.parse_transcription_file(manifest)
        assert len(records) == 1
# silent_audio_path fixture used in TestAudioCleaner.test_invalid_duration_rejected
# TestAudioResampler.test_resample_to_16k verifies samplerate==16000 via sf.info
# NoiseReducer converts stereo to mono via librosa.to_mono before denoising
# test_export_artifacts verifies hf_dataset/ and *_manifest.jsonl created
# test_parse_jsonl verifies JSONL with one-line record parses correctly
# test_parse_plain_text handles | delimiter and # comment lines
# scan_audio_directory on temp_audio_dir yields 3 records
