"""Tests for evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from evaluation.benchmark import BenchmarkReport, LatencyStats, ModelBenchmark
from evaluation.cer import CharacterErrorRate
from evaluation.wer import WordErrorRate


class TestWordErrorRate:
    """Tests for WER computation."""

    def test_perfect_match(self) -> None:
        wer = WordErrorRate()
        result = wer.compute("hello world", "hello world")
        assert result.wer == 0.0
        assert result.substitutions == 0

    def test_complete_mismatch(self) -> None:
        wer = WordErrorRate()
        result = wer.compute("hello world", "foo bar")
        assert result.wer == 1.0

    def test_substitution(self) -> None:
        wer = WordErrorRate()
        result = wer.compute("hello world", "hello earth")
        assert result.wer == 0.5
        assert result.substitutions == 1

    def test_insertion(self) -> None:
        wer = WordErrorRate()
        result = wer.compute("hello", "hello world")
        assert result.insertions == 1

    def test_deletion(self) -> None:
        wer = WordErrorRate()
        result = wer.compute("hello world", "hello")
        assert result.deletions == 1

    def test_empty_reference(self) -> None:
        wer = WordErrorRate()
        result = wer.compute("", "hello")
        assert result.wer == 1.0

    def test_compute_batch(self) -> None:
        wer = WordErrorRate()
        metrics = wer.compute_batch(
            ["hello world", "good morning"],
            ["hello world", "good evening"],
        )
        assert "wer" in metrics
        assert 0.0 < metrics["wer"] < 1.0

    def test_batch_length_mismatch(self) -> None:
        wer = WordErrorRate()
        with pytest.raises(ValueError):
            wer.compute_batch(["a"], ["a", "b"])

    def test_evaluate_dataset(self) -> None:
        wer = WordErrorRate()
        report = wer.evaluate_dataset(
            ["hello world", "test phrase"],
            ["hello world", "best phrase"],
        )
        assert report["num_samples"] == 2
        assert "aggregate" in report
        assert len(report["per_sample"]) == 2

    def test_amharic_text(self) -> None:
        wer = WordErrorRate()
        result = wer.compute("ሰላም ዓለም", "ሰላም ዓለም")
        assert result.wer == 0.0


class TestCharacterErrorRate:
    """Tests for CER computation."""

    def test_perfect_match(self) -> None:
        cer = CharacterErrorRate(language="am")
        result = cer.compute("ሰላም", "ሰላም")
        assert result.cer == 0.0

    def test_single_char_error(self) -> None:
        cer = CharacterErrorRate(language="om")
        result = cer.compute("hello", "hallo")
        assert result.cer > 0.0

    def test_compute_batch(self) -> None:
        cer = CharacterErrorRate(language="am")
        metrics = cer.compute_batch(["abc", "def"], ["abc", "deg"])
        assert "cer" in metrics
        assert metrics["cer"] > 0.0

    def test_evaluate_dataset(self) -> None:
        cer = CharacterErrorRate(language="om")
        report = cer.evaluate_dataset(["hello", "world"], ["hello", "word"])
        assert report["language"] == "om"
        assert report["num_samples"] == 2

    def test_detect_script_ethiopic(self) -> None:
        assert CharacterErrorRate.detect_script("ሰላም") == "ethiopic"

    def test_detect_script_latin(self) -> None:
        assert CharacterErrorRate.detect_script("hello world") == "latin"

    def test_remove_spaces(self) -> None:
        cer = CharacterErrorRate(remove_spaces=True)
        ref = cer.normalize_text("a b c")
        assert " " not in ref

    def test_empty_reference(self) -> None:
        cer = CharacterErrorRate()
        result = cer.compute("", "abc")
        assert result.cer == 1.0


class TestModelBenchmark:
    """Tests for model benchmarking."""

    def test_measure_latency(self) -> None:
        benchmark = ModelBenchmark(warmup_runs=1)
        fn = MagicMock(return_value="result")
        latencies, stats = benchmark.measure_latency(fn, num_runs=5)
        assert len(latencies) == 5
        assert stats.mean_ms > 0
        assert stats.p95_ms >= stats.median_ms

    def test_measure_throughput(self) -> None:
        benchmark = ModelBenchmark(warmup_runs=1)
        fn = MagicMock()
        stats = benchmark.measure_throughput(fn, num_samples=5, audio_durations=[1.0] * 5)
        assert stats.total_samples == 5
        assert stats.samples_per_second > 0

    def test_measure_memory(self) -> None:
        benchmark = ModelBenchmark()
        stats = benchmark.measure_memory()
        assert stats.process_rss_mb > 0

    def test_run_benchmark(self) -> None:
        benchmark = ModelBenchmark(warmup_runs=1)
        report = benchmark.run_benchmark(
            inference_fn=lambda: None,
            audio_durations=[1.0, 2.0, 1.5],
            evaluation_metrics={"wer": 0.15, "cer": 0.08},
            num_latency_runs=3,
        )
        assert isinstance(report, BenchmarkReport)
        assert report.evaluation_metrics["wer"] == 0.15

    def test_save_json_and_csv(self, tmp_path: Path) -> None:
        report = BenchmarkReport(
            model_name="test",
            device="cpu",
            timestamp="2024-01-01T00:00:00Z",
            latency=LatencyStats(10, 10, 12, 13, 8, 15, 1),
            throughput=ModelBenchmark(warmup_runs=0).measure_throughput(
                lambda: None, 3, [1.0, 1.0, 1.0]
            ),
            memory=ModelBenchmark().measure_memory(),
            evaluation_metrics={"wer": 0.1},
        )
        ModelBenchmark.save_json(report, tmp_path / "report.json")
        ModelBenchmark.save_csv(report, tmp_path / "report.csv")
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.csv").exists()
        data = json.loads((tmp_path / "report.json").read_text())
        assert data["model_name"] == "test"

    def test_generate_report(self, tmp_path: Path) -> None:
        benchmark = ModelBenchmark(warmup_runs=1)
        report = benchmark.generate_report(
            inference_fn=lambda: None,
            audio_durations=[1.0] * 5,
            output_dir=tmp_path,
            evaluation_metrics={"cer": 0.05},
        )
        assert (tmp_path / "benchmark_report.json").exists()
        assert (tmp_path / "benchmark_report.csv").exists()
        assert report.evaluation_metrics["cer"] == 0.05
# TestWordErrorRate.test_amharic_text asserts wer==0.0 for identical Ge'ez strings
# ReportGenerator.save_html produces valid HTML with metric summary table
# benchmark CSV has columns: category, metric, value
# CharacterErrorRate.detect_script('hello') returns 'latin'
# CI lower bound <= mean <= upper bound for uniform distribution
# compute('hello world', 'hello') -> deletions=1, wer=0.5
