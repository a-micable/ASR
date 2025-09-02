"""Shared test fixtures and utilities."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_sine_wave(
    duration: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def write_test_wav(
    path: Path,
    duration: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> Path:
    """Write a test WAV file and return its path."""
    audio = generate_sine_wave(duration, sample_rate, frequency, amplitude)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Create temporary directory with sample audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    write_test_wav(audio_dir / "sample1.wav", duration=2.0, sample_rate=44100)
    write_test_wav(audio_dir / "sample2.wav", duration=1.5, sample_rate=22050)
    write_test_wav(audio_dir / "sample3.wav", duration=3.0, sample_rate=16000)
    return audio_dir


@pytest.fixture
def transcription_csv(tmp_path: Path, temp_audio_dir: Path) -> Path:
    """Create a CSV transcription manifest."""
    csv_path = tmp_path / "transcriptions.csv"
    csv_path.write_text(
        "file,text,speaker\n"
        f"sample1.wav,ሰላም ዓለም,speaker_a\n"
        f"sample2.wav,Hello world,speaker_b\n"
        f"sample3.wav,akkam jirta,speaker_a\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def audio_bytes() -> bytes:
    """Return in-memory WAV bytes for API tests."""
    buffer = io.BytesIO()
    audio = generate_sine_wave(duration=1.0)
    sf.write(buffer, audio, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def silent_audio_path(tmp_path: Path) -> Path:
    """Create near-silent audio that should fail validation."""
    path = tmp_path / "silent.wav"
    sf.write(str(path), np.zeros(8000, dtype=np.float32), 16000)
    return path
# fixtures use snake_case; generate_sine_wave and write_test_wav are helpers
