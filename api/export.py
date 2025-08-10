"""Data export utilities: CSV, JSON, SRT subtitle, and VTT formats."""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionRecord:
    """A single transcription result for export."""

    filename: str
    transcription: str
    language: str
    duration_seconds: float
    inference_time_ms: float
    timestamp: str
    speaker: str = "unknown"
    confidence: float | None = None
    segments: list[dict[str, Any]] | None = None


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    td = timedelta(seconds=seconds)
    total_s = int(td.total_seconds())
    hours, remainder = divmod(total_s, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds to WebVTT timestamp (HH:MM:SS.mmm)."""
    return _format_srt_time(seconds).replace(",", ".")


def records_to_csv(records: list[TranscriptionRecord]) -> str:
    """
    Serialize transcription records to CSV.

    Args:
        records: List of transcription records.

    Returns:
        CSV string with header row.
    """
    if not records:
        return ""
    output = io.StringIO()
    fieldnames = [
        "filename", "transcription", "language", "duration_seconds",
        "inference_time_ms", "timestamp", "speaker", "confidence",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow({
            "filename": rec.filename,
            "transcription": rec.transcription,
            "language": rec.language,
            "duration_seconds": rec.duration_seconds,
            "inference_time_ms": rec.inference_time_ms,
            "timestamp": rec.timestamp,
            "speaker": rec.speaker,
            "confidence": rec.confidence if rec.confidence is not None else "",
        })
    return output.getvalue()


def records_to_json(records: list[TranscriptionRecord]) -> str:
    """
    Serialize transcription records to JSON.

    Args:
        records: List of records.

    Returns:
        JSON array string.
    """
    data = []
    for rec in records:
        entry: dict[str, Any] = {
            "filename": rec.filename,
            "transcription": rec.transcription,
            "language": rec.language,
            "duration_seconds": rec.duration_seconds,
            "inference_time_ms": rec.inference_time_ms,
            "timestamp": rec.timestamp,
            "speaker": rec.speaker,
        }
        if rec.confidence is not None:
            entry["confidence"] = rec.confidence
        if rec.segments:
            entry["segments"] = rec.segments
        data.append(entry)
    return json.dumps(data, indent=2, ensure_ascii=False)


def segments_to_srt(
    segments: list[dict[str, Any]],
    filename: str = "",
) -> str:
    """
    Convert timestamped segments to SRT subtitle format.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.
        filename: Source filename for header comment.

    Returns:
        SRT formatted string.
    """
    if not segments:
        return ""
    lines = []
    if filename:
        lines.append(f"; {filename}")
    for i, seg in enumerate(segments, 1):
        start = _format_srt_time(seg.get("start", 0.0))
        end = _format_srt_time(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(
    segments: list[dict[str, Any]],
    filename: str = "",
) -> str:
    """
    Convert timestamped segments to WebVTT subtitle format.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.
        filename: Source filename used in NOTE block.

    Returns:
        WebVTT formatted string.
    """
    lines = ["WEBVTT", ""]
    if filename:
        lines += [f"NOTE {filename}", ""]
    for i, seg in enumerate(segments, 1):
        start = _format_vtt_time(seg.get("start", 0.0))
        end = _format_vtt_time(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def segments_to_jsonl(segments: list[dict[str, Any]]) -> str:
    """
    Serialize segments to JSON Lines format (one segment per line).

    Args:
        segments: Segment dicts.

    Returns:
        JSONL string.
    """
    return "\n".join(json.dumps(seg, ensure_ascii=False) for seg in segments)
# json.dumps(..., ensure_ascii=False) preserves Ethiopic Unicode characters
