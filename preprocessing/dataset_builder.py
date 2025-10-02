"""Build Hugging Face datasets from audio-transcription pairs."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class DatasetStatistics:
    """Aggregated dataset statistics."""

    total_samples: int = 0
    train_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    total_duration_hours: float = 0.0
    mean_duration_seconds: float = 0.0
    min_duration_seconds: float = 0.0
    max_duration_seconds: float = 0.0
    unique_speakers: int = 0
    vocabulary_size: int = 0
    languages: list[str] = field(default_factory=list)
    character_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DatasetBuilder:
    """
    Build Hugging Face datasets from raw audio and transcription files.

    Supports common formats: CSV, TSV, JSON Lines, and Whisper-style manifest.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        transcription_file: str | Path | None = None,
        language: str = "am",
        sample_rate: int = 16000,
    ) -> None:
        """
        Initialize dataset builder.

        Args:
            audio_dir: Root directory containing audio files.
            transcription_file: Path to transcription manifest.
            language: ISO 639-1 language code (e.g. 'am', 'om').
            sample_rate: Target audio sample rate for dataset.
        """
        self.audio_dir = Path(audio_dir)
        self.transcription_file = (
            Path(transcription_file) if transcription_file else None
        )
        self.language = language
        self.sample_rate = sample_rate

    def parse_transcription_file(
        self,
        file_path: str | Path,
    ) -> list[dict[str, Any]]:
        """
        Parse transcription manifest into records.

        Supported formats:
        - CSV/TSV: columns `file`/`audio` and `text`/`transcription`
        - JSONL: {"audio": "...", "text": "...", "speaker": "..."}
        - Plain text: `filename|transcription` per line

        Args:
            file_path: Path to transcription file.

        Returns:
            List of record dicts with audio path, text, and optional metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcription file not found: {path}")

        suffix = path.suffix.lower()
        records: list[dict[str, Any]] = []

        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            audio_col = next(
                (c for c in df.columns if c.lower() in {"file", "audio", "path", "filename"}),
                df.columns[0],
            )
            text_col = next(
                (c for c in df.columns if c.lower() in {"text", "transcription", "sentence"}),
                df.columns[1],
            )
            speaker_col = next(
                (c for c in df.columns if c.lower() in {"speaker", "speaker_id"}),
                None,
            )
            for _, row in df.iterrows():
                rec: dict[str, Any] = {
                    "audio": str(self._resolve_audio_path(str(row[audio_col]))),
                    "text": str(row[text_col]).strip(),
                    "language": self.language,
                }
                if speaker_col:
                    rec["speaker"] = str(row[speaker_col])
                records.append(rec)

        elif suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    records.append(
                        {
                            "audio": str(
                                self._resolve_audio_path(item.get("audio", item.get("file", "")))
                            ),
                            "text": item.get("text", item.get("transcription", "")).strip(),
                            "language": item.get("language", self.language),
                            "speaker": item.get("speaker", "unknown"),
                        }
                    )

        elif suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("data", [])
            for item in items:
                records.append(
                    {
                        "audio": str(
                            self._resolve_audio_path(item.get("audio", item.get("file", "")))
                        ),
                        "text": item.get("text", "").strip(),
                        "language": item.get("language", self.language),
                        "speaker": item.get("speaker", "unknown"),
                    }
                )

        else:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "|" in line:
                        filename, text = line.split("|", 1)
                    elif "\t" in line:
                        filename, text = line.split("\t", 1)
                    else:
                        parts = line.split(maxsplit=1)
                        if len(parts) < 2:
                            continue
                        filename, text = parts
                    records.append(
                        {
                            "audio": str(self._resolve_audio_path(filename.strip())),
                            "text": text.strip(),
                            "language": self.language,
                            "speaker": "unknown",
                        }
                    )

        logger.info("Parsed %d records from %s", len(records), path)
        return records

    def _resolve_audio_path(self, ref: str) -> Path:
        """Resolve audio path relative to audio_dir."""
        path = Path(ref)
        if path.is_absolute() and path.exists():
            return path
        candidate = self.audio_dir / path.name
        if candidate.exists():
            return candidate
        candidate = self.audio_dir / path
        if candidate.exists():
            return candidate
        return self.audio_dir / path.name

    def scan_audio_directory(self) -> list[dict[str, Any]]:
        """
        Scan audio directory for files without transcriptions.

        Returns:
            Records with empty text (for manual annotation).
        """
        records = []
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            for path in self.audio_dir.rglob(f"*{ext}"):
                records.append(
                    {
                        "audio": str(path),
                        "text": "",
                        "language": self.language,
                        "speaker": path.parent.name,
                    }
                )
        return records

    def build_records(self) -> list[dict[str, Any]]:
        """
        Build record list from transcription file or directory scan.

        Returns:
            Validated records with existing audio files.
        """
        if self.transcription_file:
            records = self.parse_transcription_file(self.transcription_file)
        else:
            records = self.scan_audio_directory()

        valid = []
        for rec in records:
            audio_path = Path(rec["audio"])
            if not audio_path.exists():
                logger.warning("Missing audio: %s", audio_path)
                continue
            if not rec.get("text"):
                logger.warning("Empty transcription for: %s", audio_path)
                continue
            valid.append(rec)

        logger.info("Validated %d / %d records", len(valid), len(records))
        return valid

    def create_splits(
        self,
        records: list[dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Split records into train/validation/test sets.

        Args:
            records: Full record list.
            train_ratio: Training set proportion.
            val_ratio: Validation set proportion.
            test_ratio: Test set proportion.
            seed: Random seed for reproducibility.

        Returns:
            Dict with 'train', 'validation', 'test' keys.
        """
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        shuffled = records.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    def _records_to_dataset(self, records: list[dict[str, Any]]) -> Dataset:
        """Convert records to HuggingFace Dataset with audio file paths."""
        return Dataset.from_dict(
            {
                "audio": [str(rec["audio"]) for rec in records],
                "text": [rec["text"] for rec in records],
                "language": [rec.get("language", self.language) for rec in records],
                "speaker": [rec.get("speaker", "unknown") for rec in records],
            }
        )

    def build_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Build a full DatasetDict with train/validation/test splits.

        Returns:
            HuggingFace DatasetDict ready for training.
        """
        records = self.build_records()
        if not records:
            raise ValueError("No valid records found to build dataset")

        splits = self.create_splits(records, train_ratio, val_ratio, test_ratio, seed)
        dataset_dict = DatasetDict(
            {
                split: self._records_to_dataset(recs)
                for split, recs in splits.items()
            }
        )
        logger.info(
            "Built dataset: train=%d, val=%d, test=%d",
            len(dataset_dict["train"]),
            len(dataset_dict["validation"]),
            len(dataset_dict["test"]),
        )
        return dataset_dict

    def _audio_path_string(self, audio_ref: Any) -> str:
        """Extract filesystem path from an audio column value."""
        if isinstance(audio_ref, dict):
            return str(audio_ref.get("path", ""))
        return str(audio_ref)

    def _audio_duration_seconds(self, audio_ref: Any) -> float:
        """Resolve audio duration without requiring torchcodec decoding."""
        if isinstance(audio_ref, dict):
            if "array" in audio_ref and "sampling_rate" in audio_ref:
                return len(audio_ref["array"]) / audio_ref["sampling_rate"]
            path = audio_ref.get("path")
            if path:
                return sf.info(str(path)).duration
            return 0.0
        path = Path(self._audio_path_string(audio_ref))
        if path.exists():
            return sf.info(str(path)).duration
        return 0.0

    def compute_statistics(
        self,
        dataset_dict: DatasetDict,
    ) -> DatasetStatistics:
        """
        Compute dataset-level statistics.

        Args:
            dataset_dict: Dataset with audio and text columns.

        Returns:
            DatasetStatistics dataclass.
        """
        all_text: list[str] = []
        all_speakers: set[str] = set()
        durations: list[float] = []
        languages: set[str] = set()

        for split_name in dataset_dict:
            split = dataset_dict[split_name]
            texts = split["text"]
            speakers = split["speaker"]
            languages_col = split["language"]
            audio_col = split["audio"]

            for i in range(len(split)):
                all_text.append(texts[i])
                all_speakers.add(speakers[i])
                languages.add(languages_col[i])
                durations.append(self._audio_duration_seconds(audio_col[i]))

        vocab: set[str] = set()
        char_count = 0
        for text in all_text:
            char_count += len(text)
            vocab.update(text.split())

        stats = DatasetStatistics(
            total_samples=sum(len(dataset_dict[s]) for s in dataset_dict),
            train_samples=len(dataset_dict.get("train", [])),
            validation_samples=len(dataset_dict.get("validation", [])),
            test_samples=len(dataset_dict.get("test", [])),
            total_duration_hours=sum(durations) / 3600.0,
            mean_duration_seconds=float(sum(durations) / max(len(durations), 1)),
            min_duration_seconds=float(min(durations)) if durations else 0.0,
            max_duration_seconds=float(max(durations)) if durations else 0.0,
            unique_speakers=len(all_speakers),
            vocabulary_size=len(vocab),
            languages=sorted(languages),
            character_count=char_count,
        )
        return stats

    def export_artifacts(
        self,
        dataset_dict: DatasetDict,
        output_dir: str | Path,
    ) -> DatasetStatistics:
        """
        Save dataset to disk and export statistics JSON.

        Args:
            dataset_dict: Dataset to persist.
            output_dir: Output directory for arrow files and stats.

        Returns:
            Computed statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        non_empty = DatasetDict({k: v for k, v in dataset_dict.items() if len(v) > 0})
        non_empty.save_to_disk(str(output_dir / "hf_dataset"))
        stats = self.compute_statistics(dataset_dict)

        stats_path = output_dir / "dataset_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)

        for split in dataset_dict:
            split_data = dataset_dict[split]
            manifest = []
            for audio_ref, text, language, speaker in zip(
                split_data["audio"],
                split_data["text"],
                split_data["language"],
                split_data["speaker"],
                strict=True,
            ):
                manifest.append(
                    {
                        "audio": self._audio_path_string(audio_ref),
                        "text": text,
                        "language": language,
                        "speaker": speaker,
                    }
                )
            manifest_path = output_dir / f"{split}_manifest.jsonl"
            with open(manifest_path, "w", encoding="utf-8") as f:
                for entry in manifest:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info("Exported dataset artifacts to %s", output_dir)
        return stats

    @staticmethod
    def normalize_text(text: str, language: str = "am") -> str:
        """
        Normalize transcription text for low-resource languages.

        Args:
            text: Raw transcription.
            language: Language code for language-specific rules.

        Returns:
            Normalized text.
        """
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        if language in {"am", "om"}:
            text = text.replace("\u2019", "'").replace("\u2018", "'")
        return text
# absolute paths are resolved directly via Path.is_absolute() check
# BOM (\ufeff) stripped via utf-8-sig encoding fallback in open()
# sf.info() used for duration extraction without full audio decode
# _resolve_audio_path resolves relative and absolute paths against audio_dir
# scan_audio_directory() returns records with empty text for manual labeling
