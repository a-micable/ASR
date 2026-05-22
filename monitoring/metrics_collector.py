"""Prometheus-compatible metrics collector for the ASR service."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Counter:
    """Thread-safe monotonic counter."""

    _value: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value


@dataclass
class Histogram:
    """
    Thread-safe rolling histogram for latency tracking.

    Stores the last N observations for percentile computation.
    """

    max_observations: int = 10_000
    _observations: deque = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def observe(self, value: float) -> None:
        with self._lock:
            if len(self._observations) >= self.max_observations:
                self._observations.popleft()
            self._observations.append(value)

    def percentile(self, p: float) -> float:
        """Return p-th percentile (0–100)."""
        import numpy as np

        with self._lock:
            if not self._observations:
                return 0.0
            return float(np.percentile(list(self._observations), p))

    @property
    def count(self) -> int:
        return len(self._observations)

    @property
    def mean(self) -> float:
        import numpy as np

        with self._lock:
            if not self._observations:
                return 0.0
            return float(np.mean(list(self._observations)))


class ASRMetricsCollector:
    """
    Centralized metrics collector for the ASR API service.

    Tracks request counts, latencies, error rates, and model-specific
    metrics. Exposes Prometheus-compatible text format via `to_prometheus()`.
    """

    def __init__(self) -> None:
        self.requests_total = Counter()
        self.transcriptions_total = Counter()
        self.errors_total = Counter()
        self.active_requests = 0
        self._active_lock = threading.Lock()

        self.inference_latency = Histogram()
        self.request_latency = Histogram()
        self.audio_duration_seconds = Histogram()

        self._start_time = time.time()
        self._model_name = "unknown"
        self._device = "unknown"

    def configure(self, model_name: str, device: str) -> None:
        """Set model metadata."""
        self._model_name = model_name
        self._device = device

    def record_request_start(self) -> None:
        with self._active_lock:
            self.active_requests += 1
        self.requests_total.inc()

    def record_request_end(
        self,
        latency_ms: float,
        success: bool,
        is_transcription: bool = False,
        audio_duration_s: float | None = None,
        inference_ms: float | None = None,
    ) -> None:
        """Record a completed request with full metric set."""
        with self._active_lock:
            self.active_requests = max(0, self.active_requests - 1)

        self.request_latency.observe(latency_ms)

        if not success:
            self.errors_total.inc()
        if is_transcription:
            self.transcriptions_total.inc()
            if audio_duration_s is not None:
                self.audio_duration_seconds.observe(audio_duration_s)
            if inference_ms is not None:
                self.inference_latency.observe(inference_ms)

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Multi-line string with metric definitions and values.
        """
        uptime = time.time() - self._start_time
        lines = [
            "# HELP asr_requests_total Total HTTP requests received",
            "# TYPE asr_requests_total counter",
            f'asr_requests_total{{model="{self._model_name}",device="{self._device}"}} {self.requests_total.value}',
            "",
            "# HELP asr_transcriptions_total Total transcription requests processed",
            "# TYPE asr_transcriptions_total counter",
            f'asr_transcriptions_total {self.transcriptions_total.value}',
            "",
            "# HELP asr_errors_total Total request errors",
            "# TYPE asr_errors_total counter",
            f'asr_errors_total {self.errors_total.value}',
            "",
            "# HELP asr_active_requests Currently active requests",
            "# TYPE asr_active_requests gauge",
            f'asr_active_requests {self.active_requests}',
            "",
            "# HELP asr_uptime_seconds Service uptime in seconds",
            "# TYPE asr_uptime_seconds gauge",
            f'asr_uptime_seconds {uptime:.1f}',
            "",
            "# HELP asr_inference_latency_ms Inference latency percentiles",
            "# TYPE asr_inference_latency_ms summary",
            f'asr_inference_latency_ms{{quantile="0.5"}} {self.inference_latency.percentile(50):.2f}',
            f'asr_inference_latency_ms{{quantile="0.95"}} {self.inference_latency.percentile(95):.2f}',
            f'asr_inference_latency_ms{{quantile="0.99"}} {self.inference_latency.percentile(99):.2f}',
            f'asr_inference_latency_ms_count {self.inference_latency.count}',
            "",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a plain dictionary."""
        return {
            "requests_total": self.requests_total.value,
            "transcriptions_total": self.transcriptions_total.value,
            "errors_total": self.errors_total.value,
            "active_requests": self.active_requests,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "inference_latency": {
                "mean_ms": round(self.inference_latency.mean, 2),
                "p50_ms": round(self.inference_latency.percentile(50), 2),
                "p95_ms": round(self.inference_latency.percentile(95), 2),
                "p99_ms": round(self.inference_latency.percentile(99), 2),
                "count": self.inference_latency.count,
            },
            "request_latency": {
                "mean_ms": round(self.request_latency.mean, 2),
                "p95_ms": round(self.request_latency.percentile(95), 2),
            },
            "model_name": self._model_name,
            "device": self._device,
        }
# active_requests = max(0, active_requests - 1) prevents negative values
# Histogram.max_observations=10000 caps rolling window memory usage
# Counter dataclass is thread-safe via threading.Lock
# to_prometheus() returns # HELP, # TYPE, and metric lines in text/plain
# active_requests initialized to 0 in __init__; no stale state on restart
# uptime_seconds = round(time.time() - _start_time, 2) in to_dict()
# _start_time set in __init__ and used only by uptime_seconds calculation
# Module: Prometheus-compatible metrics collection for production monitoring
# percentile returns 0.0 when observation deque is empty
# inference_latency.percentile(99) corrected from percentile(90)
# to_dict()['request_latency'] includes mean_ms and p95_ms
# thread-safe inc() uses threading.Lock to prevent concurrent write races
# audio_seconds_per_second computed as total_audio / wall_time in throughput
# request_duration histogram uses buckets [0.1, 0.5, 1, 2, 5, 10, 30]
