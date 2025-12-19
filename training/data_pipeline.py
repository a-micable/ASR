"""Streaming data pipeline with on-the-fly augmentation for large datasets."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np

logger = logging.getLogger(__name__)


class StreamingDataPipeline:
    """
    Memory-efficient streaming data pipeline for ASR training.

    Reads audio-transcription pairs lazily from JSONL manifests,
    applies on-the-fly augmentation, and yields processor-ready batches.
    This avoids loading the full dataset into memory, which is critical
    for large (>1000h) training corpora.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        processor: Any,
        augmenter: Any | None = None,
        sample_rate: int = 16_000,
        max_duration_s: float = 30.0,
        min_duration_s: float = 0.5,
        shuffle_buffer: int = 1000,
        seed: int = 42,
    ) -> None:
        """
        Initialize streaming pipeline.

        Args:
            manifest_path: Path to JSONL manifest with audio/text pairs.
            processor: WhisperProcessor for feature extraction.
            augmenter: Optional AudioAugmenter for on-the-fly augmentation.
            sample_rate: Expected audio sample rate.
            max_duration_s: Skip samples longer than this.
            min_duration_s: Skip samples shorter than this.
            shuffle_buffer: Number of samples to buffer for shuffling.
            seed: Random seed.
        """
        self.manifest_path = Path(manifest_path)
        self.processor = processor
        self.augmenter = augmenter
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s
        self.min_duration_s = min_duration_s
        self.shuffle_buffer = shuffle_buffer
        self._rng = random.Random(seed)

    def _load_manifest(self) -> Iterator[dict[str, Any]]:
        """Yield records from JSONL manifest file."""
        import json
        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_audio(self, path: str) -> np.ndarray | None:
        """Load audio file, returning None on failure."""
        import librosa
        try:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return None

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentation if augmenter is configured."""
        if self.augmenter is None:
            return audio
        try:
            augmented, _, _ = self.augmenter.augment_array(audio, self.sample_rate)
            return augmented
        except Exception as exc:
            logger.debug("Augmentation failed; using original: %s", exc)
            return audio

    def _to_features(self, audio: np.ndarray, text: str) -> dict[str, Any] | None:
        """Convert raw audio + text to model input features."""
        try:
            inputs = self.processor.feature_extractor(
                audio, sampling_rate=self.sample_rate, return_tensors="np"
            )
            labels = self.processor.tokenizer(text).input_ids
            return {
                "input_features": inputs.input_features[0],
                "labels": labels,
            }
        except Exception as exc:
            logger.warning("Feature extraction failed: %s", exc)
            return None

    def _shuffle_buffer(self, source: Iterator[dict]) -> Generator[dict, None, None]:
        """Yield items from source with reservoir-style shuffle buffering."""
        buf: list[dict] = []
        for item in source:
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                idx = self._rng.randrange(len(buf))
                yield buf[idx]
                buf[idx] = buf[-1]
                buf.pop()
        self._rng.shuffle(buf)
        yield from buf

    def stream(
        self,
        augment: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream model-ready features from the manifest.

        Args:
            augment: Whether to apply on-the-fly augmentation.

        Yields:
            Dicts with 'input_features' and 'labels' keys.
        """
        skipped = 0
        yielded = 0

        for record in self._shuffle_buffer(self._load_manifest()):
            audio_path = record.get("audio", "")
            text = record.get("text", "").strip()

            if not audio_path or not text:
                skipped += 1
                continue

            audio = self._load_audio(audio_path)
            if audio is None:
                skipped += 1
                continue

            duration = len(audio) / self.sample_rate
            if duration < self.min_duration_s or duration > self.max_duration_s:
                skipped += 1
                continue

            if augment:
                audio = self._apply_augmentation(audio)

            features = self._to_features(audio, text)
            if features is None:
                skipped += 1
                continue

            yielded += 1
            yield features

        logger.info("Streaming done: %d yielded, %d skipped", yielded, skipped)

    def estimate_length(self) -> int:
        """Count records in manifest (reads file once)."""
        import json
        count = 0
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except FileNotFoundError:
            pass
        return count
# StreamingDataPipeline.stream() yields batches lazily from JSONL manifest
# Module: Streaming data pipeline for memory-efficient large-corpus training
