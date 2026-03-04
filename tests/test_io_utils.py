"""Tests for preprocessing/io_utils — the boilerplate elimination module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from preprocessing.io_utils import (
    auto_output_path,
    ensure_output_dir,
    load_audio,
    save_audio,
)
from tests.conftest import generate_sine_wave, write_test_wav


class TestLoadAudio:
    def test_loads_mono(self, temp_audio_dir: Path) -> None:
        audio, sr = load_audio(temp_audio_dir / "sample1.wav", mono=True)
        assert audio.ndim == 1
        assert sr > 0

    def test_loads_stereo_preserves_channels(self, tmp_path: Path) -> None:
        stereo = np.stack([generate_sine_wave(), generate_sine_wave()])
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo.T, 16000)
        audio, sr = load_audio(path, mono=False)
        assert sr == 16000

    def test_resamples_on_load(self, temp_audio_dir: Path) -> None:
        # sample1.wav was written at 44100; request 16000
        audio, sr = load_audio(temp_audio_dir / "sample1.wav", mono=True, target_sr=16000)
        assert sr == 16000

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_path / "nonexistent.wav")

    def test_corrupt_file_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not audio data at all")
        with pytest.raises(ValueError):
            load_audio(bad)


class TestSaveAudio:
    def test_creates_file(self, tmp_path: Path) -> None:
        audio = generate_sine_wave()
        out = tmp_path / "sub" / "out.wav"
        result = save_audio(audio, 16000, out)
        assert result.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        audio = generate_sine_wave()
        deep = tmp_path / "a" / "b" / "c" / "out.wav"
        save_audio(audio, 16000, deep)
        assert deep.exists()

    def test_round_trip(self, tmp_path: Path) -> None:
        original = generate_sine_wave(amplitude=0.5)
        out = tmp_path / "rt.wav"
        save_audio(original, 16000, out)
        loaded, sr = sf.read(str(out))
        assert sr == 16000
        assert len(loaded) == len(original)

    def test_returns_path(self, tmp_path: Path) -> None:
        audio = generate_sine_wave()
        out = tmp_path / "result.wav"
        result = save_audio(audio, 16000, out)
        assert isinstance(result, Path)
        assert result == out


class TestEnsureOutputDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new" / "deep"
        result = ensure_output_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_directory_no_error(self, tmp_path: Path) -> None:
        result = ensure_output_dir(tmp_path)
        assert result == tmp_path

    def test_returns_path_object(self, tmp_path: Path) -> None:
        result = ensure_output_dir(str(tmp_path / "x"))
        assert isinstance(result, Path)


class TestAutoOutputPath:
    def test_same_directory_by_default(self) -> None:
        path = auto_output_path("data/raw/speech.wav", "_clean")
        assert path.parent == Path("data/raw")
        assert path.name == "speech_clean.wav"

    def test_override_directory(self) -> None:
        path = auto_output_path("data/raw/speech.wav", "_16k", "data/processed")
        assert path.parent == Path("data/processed")
        assert path.name == "speech_16k.wav"

    def test_custom_extension(self) -> None:
        path = auto_output_path("audio/clip.wav", "_aug0", extension="flac")
        assert path.suffix == ".flac"

    def test_suffix_appended_to_stem(self) -> None:
        path = auto_output_path("audio/clip.wav", "_denoised")
        assert "denoised" in path.stem

    def test_path_input_accepted(self) -> None:
        path = auto_output_path(Path("data/file.wav"), "_clean")
        assert isinstance(path, Path)
