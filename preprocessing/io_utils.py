"""
Shared I/O utilities for audio preprocessing modules.

Eliminates repeated boilerplate across audio_cleaner, noise_reducer,
augmentation, and resampler: file loading, saving, and output path setup.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Default write format for all preprocessing output
_WRITE_SUBTYPE = "PCM_16"


def load_audio(
    path: str | Path,
    mono: bool = True,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load audio from disk with unified error handling.

    Replaces repeated ``librosa.load(str(path), sr=None, mono=True/False)``
    calls scattered across audio_cleaner, noise_reducer, augmentation,
    and resampler.

    Args:
        path: Path to audio file (WAV, FLAC, MP3, OGG, M4A).
        mono: Mix down to mono when True.
        target_sr: Resample to this rate; preserves original rate when None.

    Returns:
        (waveform, sample_rate) tuple.

    Raises:
        ValueError: When the file cannot be decoded.
        FileNotFoundError: When the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)
        return audio, sr
    except Exception as exc:
        raise ValueError(f"Failed to decode audio {path.name}: {exc}") from exc


def save_audio(
    audio: np.ndarray,
    sr: int,
    output_path: str | Path,
    subtype: str = _WRITE_SUBTYPE,
) -> Path:
    """
    Write audio to disk, creating parent directories as needed.

    Replaces the repeated pattern:
    ``output_path.parent.mkdir(parents=True, exist_ok=True)``
    ``sf.write(str(output_path), audio, sr, subtype="PCM_16")``

    Args:
        audio: Mono float32 waveform.
        sr: Sample rate in Hz.
        output_path: Destination file path.
        subtype: Soundfile subtype; defaults to PCM_16.

    Returns:
        Resolved output Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype=subtype)
    return output_path


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Resolve and create an output directory.

    Replaces the repeated two-liner:
    ``output_dir = Path(output_dir)``
    ``output_dir.mkdir(parents=True, exist_ok=True)``

    Args:
        output_dir: Directory path to create.

    Returns:
        Resolved Path object.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def auto_output_path(
    input_path: str | Path,
    suffix: str,
    output_dir: str | Path | None = None,
    extension: str = "wav",
) -> Path:
    """
    Derive an output path from an input path and a suffix string.

    Replaces repeated patterns like:
    ``if output_path is None:``
    ``    output_path = input_path.parent / f"{input_path.stem}_clean.wav"``

    Args:
        input_path: Source file path.
        suffix: String appended to stem, e.g. ``"_clean"``, ``"_16k"``.
        output_dir: Override directory; uses input parent when None.
        extension: Output file extension without dot.

    Returns:
        Computed output Path (parent directory not yet created).

    Example:
        >>> auto_output_path("data/raw/speech.wav", "_clean")
        PosixPath('data/raw/speech_clean.wav')
        >>> auto_output_path("data/raw/speech.wav", "_16k", "data/processed")
        PosixPath('data/processed/speech_16k.wav')
    """
    input_path = Path(input_path)
    stem = f"{input_path.stem}{suffix}"
    filename = f"{stem}.{extension}"
    directory = Path(output_dir) if output_dir else input_path.parent
    return directory / filename
