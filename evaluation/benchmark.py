"""Model benchmarking: latency, throughput, memory, and report generation."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Inference latency statistics."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    samples_per_second: float
    audio_seconds_per_second: float
    total_samples: int
    total_audio_seconds: float
    wall_time_seconds: float


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    process_rss_mb: float
    process_vms_mb: float
    gpu_allocated_mb: float | None
    gpu_reserved_mb: float | None
    gpu_max_allocated_mb: float | None


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    model_name: str
    device: str
    timestamp: str
    latency: LatencyStats
    throughput: ThroughputStats
    memory: MemoryStats
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary."""
        return asdict(self)


class ModelBenchmark:
    """
    Benchmark ASR model inference performance.

    Measures latency, throughput, memory usage, and generates JSON/CSV reports.
    """

    def __init__(
        self,
        model_name: str = "whisper",
        device: str | None = None,
        warmup_runs: int = 3,
    ) -> None:
        """
        Initialize benchmark runner.

        Args:
            model_name: Identifier for reports.
            device: Torch device string; auto-detected if None.
            warmup_runs: Number of warmup iterations before timing.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.warmup_runs = warmup_runs

    def measure_latency(
        self,
        inference_fn: Callable[[], Any],
        num_runs: int = 50,
    ) -> tuple[list[float], LatencyStats]:
        """
        Measure inference latency over multiple runs.

        Args:
            inference_fn: Zero-argument callable performing one inference.
            num_runs: Number of timed runs.

        Returns:
            Raw latencies (ms) and aggregated stats.
        """
        for _ in range(self.warmup_runs):
            inference_fn()

        latencies: list[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            inference_fn()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        arr = np.array(latencies)
        stats = LatencyStats(
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
        )
        return latencies, stats

    def measure_throughput(
        self,
        inference_fn: Callable[[], Any],
        num_samples: int,
        audio_durations: list[float],
    ) -> ThroughputStats:
        """
        Measure batch throughput.

        Args:
            inference_fn: Single-sample inference callable.
            num_samples: Number of samples to process.
            audio_durations: Duration in seconds for each sample.

        Returns:
            ThroughputStats.
        """
        for _ in range(self.warmup_runs):
            inference_fn()

        start = time.perf_counter()
        for _ in range(num_samples):
            inference_fn()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - start

        total_audio = sum(audio_durations[:num_samples])
        return ThroughputStats(
            samples_per_second=num_samples / wall_time,
            audio_seconds_per_second=total_audio / wall_time,
            total_samples=num_samples,
            total_audio_seconds=total_audio,
            wall_time_seconds=wall_time,
        )

    def measure_memory(self) -> MemoryStats:
        """
        Capture current process and GPU memory usage.

        Returns:
            MemoryStats snapshot.
        """
        process = psutil.Process()
        mem = process.memory_info()

        gpu_allocated = gpu_reserved = gpu_max = None
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
            gpu_max = torch.cuda.max_memory_allocated() / (1024**2)

        return MemoryStats(
            process_rss_mb=mem.rss / (1024**2),
            process_vms_mb=mem.vms / (1024**2),
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_max_allocated_mb=gpu_max,
        )

    def run_benchmark(
        self,
        inference_fn: Callable[[], Any],
        audio_durations: list[float],
        evaluation_metrics: dict[str, float] | None = None,
        num_latency_runs: int = 50,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkReport:
        """
        Run full benchmark suite.

        Args:
            inference_fn: Inference callable.
            audio_durations: Per-sample audio durations in seconds.
            evaluation_metrics: Optional WER/CER metrics to include.
            num_latency_runs: Latency measurement iterations.
            metadata: Additional metadata for report.

        Returns:
            Complete BenchmarkReport.
        """
        _, latency_stats = self.measure_latency(inference_fn, num_latency_runs)
        throughput_stats = self.measure_throughput(
            inference_fn,
            num_samples=min(len(audio_durations), 20),
            audio_durations=audio_durations,
        )
        memory_stats = self.measure_memory()

        report = BenchmarkReport(
            model_name=self.model_name,
            device=self.device,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency=latency_stats,
            throughput=throughput_stats,
            memory=memory_stats,
            evaluation_metrics=evaluation_metrics or {},
            metadata=metadata or {},
        )

        logger.info(
            "Benchmark complete | mean latency=%.1fms | throughput=%.1f samples/s",
            latency_stats.mean_ms,
            throughput_stats.samples_per_second,
        )
        return report

    @staticmethod
    def save_json(report: BenchmarkReport, path: str | Path) -> None:
        """
        Save report as JSON.

        Args:
            report: Benchmark report.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved benchmark JSON: %s", path)

    @staticmethod
    def save_csv(report: BenchmarkReport, path: str | Path) -> None:
        """
        Save report summary as CSV.

        Args:
            report: Benchmark report.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()

        rows = [
            {"category": "latency", "metric": k, "value": v}
            for k, v in data["latency"].items()
        ]
        rows += [
            {"category": "throughput", "metric": k, "value": v}
            for k, v in data["throughput"].items()
        ]
        rows += [
            {"category": "memory", "metric": k, "value": v}
            for k, v in data["memory"].items()
            if v is not None
        ]
        for k, v in data.get("evaluation_metrics", {}).items():
            rows.append({"category": "evaluation", "metric": k, "value": v})

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved benchmark CSV: %s", path)

    def generate_report(
        self,
        inference_fn: Callable[[], Any],
        audio_durations: list[float],
        output_dir: str | Path,
        evaluation_metrics: dict[str, float] | None = None,
    ) -> BenchmarkReport:
        """
        Run benchmark and save JSON + CSV artifacts.

        Args:
            inference_fn: Inference callable.
            audio_durations: Sample durations.
            output_dir: Directory for output files.
            evaluation_metrics: Optional eval metrics.

        Returns:
            BenchmarkReport.
        """
        output_dir = Path(output_dir)
        report = self.run_benchmark(
            inference_fn,
            audio_durations,
            evaluation_metrics=evaluation_metrics,
        )
        self.save_json(report, output_dir / "benchmark_report.json")
        self.save_csv(report, output_dir / "benchmark_report.csv")
        return report
# MemoryStats.gpu_allocated_mb is Optional[float] for CPU-only hosts
# to_dict() uses dataclasses.asdict() for deep serialization
# measure_memory() called after inference to capture peak GPU allocation
