"""Tests for transcription export formats."""

from __future__ import annotations

import json

import pytest

from api.export import (
    TranscriptionRecord,
    records_to_csv,
    records_to_json,
    segments_to_jsonl,
    segments_to_srt,
    segments_to_vtt,
)


def _sample_record(filename: str = "test.wav") -> TranscriptionRecord:
    return TranscriptionRecord(
        filename=filename,
        transcription="ሰላም ዓለም",
        language="am",
        duration_seconds=2.5,
        inference_time_ms=150.0,
        timestamp="2024-06-01T10:00:00Z",
        speaker="speaker_a",
        confidence=0.95,
    )


def _sample_segments() -> list[dict]:
    return [
        {"start": 0.0, "end": 1.5, "text": "ሰላም"},
        {"start": 1.5, "end": 3.0, "text": "ዓለም"},
    ]


class TestRecordsToCSV:
    def test_basic_output(self) -> None:
        records = [_sample_record()]
        csv_out = records_to_csv(records)
        assert "filename" in csv_out
        assert "ሰላም ዓለም" in csv_out

    def test_empty_list(self) -> None:
        assert records_to_csv([]) == ""

    def test_has_header_row(self) -> None:
        records = [_sample_record()]
        csv_out = records_to_csv(records)
        lines = csv_out.strip().split("\n")
        assert len(lines) == 2  # header + 1 record

    def test_multiple_records(self) -> None:
        records = [_sample_record("a.wav"), _sample_record("b.wav")]
        csv_out = records_to_csv(records)
        lines = csv_out.strip().split("\n")
        assert len(lines) == 3

    def test_null_confidence(self) -> None:
        rec = _sample_record()
        rec.confidence = None
        csv_out = records_to_csv([rec])
        assert csv_out  # should not crash


class TestRecordsToJSON:
    def test_valid_json(self) -> None:
        records = [_sample_record()]
        out = records_to_json(records)
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_fields_present(self) -> None:
        records = [_sample_record()]
        data = json.loads(records_to_json(records))
        assert data[0]["transcription"] == "ሰላም ዓለም"
        assert data[0]["language"] == "am"
        assert data[0]["confidence"] == 0.95

    def test_with_segments(self) -> None:
        rec = _sample_record()
        rec.segments = [{"start": 0.0, "end": 1.0, "text": "test"}]
        data = json.loads(records_to_json([rec]))
        assert "segments" in data[0]


class TestSegmentsToSRT:
    def test_basic_srt(self) -> None:
        srt = segments_to_srt(_sample_segments())
        assert "00:00:00,000" in srt or "00:00:00" in srt
        assert "ሰላም" in srt

    def test_empty_segments(self) -> None:
        assert segments_to_srt([]) == ""

    def test_sequence_numbers(self) -> None:
        srt = segments_to_srt(_sample_segments())
        assert "1\n" in srt
        assert "2\n" in srt

    def test_with_filename(self) -> None:
        srt = segments_to_srt(_sample_segments(), filename="audio.wav")
        assert "audio.wav" in srt

    def test_arrow_separator(self) -> None:
        srt = segments_to_srt(_sample_segments())
        assert "-->" in srt


class TestSegmentsToVTT:
    def test_starts_with_webvtt(self) -> None:
        vtt = segments_to_vtt(_sample_segments())
        assert vtt.startswith("WEBVTT")

    def test_uses_dot_milliseconds(self) -> None:
        vtt = segments_to_vtt(_sample_segments())
        # VTT uses . not ,
        assert "00:00:00.000" in vtt

    def test_empty_segments(self) -> None:
        vtt = segments_to_vtt([])
        assert vtt.startswith("WEBVTT")

    def test_with_filename(self) -> None:
        vtt = segments_to_vtt(_sample_segments(), filename="my.wav")
        assert "my.wav" in vtt


class TestSegmentsToJSONL:
    def test_one_line_per_segment(self) -> None:
        jsonl = segments_to_jsonl(_sample_segments())
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

    def test_valid_json_per_line(self) -> None:
        jsonl = segments_to_jsonl(_sample_segments())
        for line in jsonl.strip().split("\n"):
            data = json.loads(line)
            assert "text" in data

    def test_empty(self) -> None:
        assert segments_to_jsonl([]) == ""
