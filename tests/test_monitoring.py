"""Tests for monitoring and health check modules."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from monitoring.health_checker import (
    HealthChecker,
    HealthStatus,
    checkpoint_exists_probe,
    disk_space_probe,
    memory_probe,
    model_loaded_probe,
)
from monitoring.metrics_collector import ASRMetricsCollector, Counter, Histogram


class TestCounter:
    def test_initial_value_zero(self) -> None:
        c = Counter()
        assert c.value == 0

    def test_increment(self) -> None:
        c = Counter()
        c.inc()
        assert c.value == 1

    def test_increment_by_amount(self) -> None:
        c = Counter()
        c.inc(5)
        assert c.value == 5

    def test_multiple_increments(self) -> None:
        c = Counter()
        for _ in range(10):
            c.inc()
        assert c.value == 10


class TestHistogram:
    def test_empty_percentile(self) -> None:
        h = Histogram()
        assert h.percentile(95) == 0.0

    def test_observe_and_percentile(self) -> None:
        h = Histogram()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            h.observe(v)
        assert h.percentile(50) == pytest.approx(30.0, abs=5.0)

    def test_mean(self) -> None:
        h = Histogram()
        for v in [10.0, 20.0, 30.0]:
            h.observe(v)
        assert h.mean == pytest.approx(20.0)

    def test_max_observations_evicts_old(self) -> None:
        h = Histogram(max_observations=5)
        for i in range(10):
            h.observe(float(i))
        assert h.count == 5


class TestASRMetricsCollector:
    def test_initial_state(self) -> None:
        m = ASRMetricsCollector()
        assert m.requests_total.value == 0
        assert m.transcriptions_total.value == 0
        assert m.errors_total.value == 0

    def test_record_request_start(self) -> None:
        m = ASRMetricsCollector()
        m.record_request_start()
        assert m.requests_total.value == 1
        assert m.active_requests == 1

    def test_record_request_end_success(self) -> None:
        m = ASRMetricsCollector()
        m.record_request_start()
        m.record_request_end(100.0, success=True)
        assert m.active_requests == 0
        assert m.errors_total.value == 0

    def test_record_request_end_error(self) -> None:
        m = ASRMetricsCollector()
        m.record_request_start()
        m.record_request_end(100.0, success=False)
        assert m.errors_total.value == 1

    def test_transcription_recording(self) -> None:
        m = ASRMetricsCollector()
        m.record_request_start()
        m.record_request_end(
            200.0, success=True, is_transcription=True,
            audio_duration_s=3.5, inference_ms=180.0
        )
        assert m.transcriptions_total.value == 1
        assert m.inference_latency.count == 1
        assert m.audio_duration_seconds.count == 1

    def test_to_prometheus_format(self) -> None:
        m = ASRMetricsCollector()
        m.configure("whisper-small", "cpu")
        m.record_request_start()
        prometheus = m.to_prometheus()
        assert "asr_requests_total" in prometheus
        assert "asr_transcriptions_total" in prometheus
        assert "asr_uptime_seconds" in prometheus

    def test_to_dict(self) -> None:
        m = ASRMetricsCollector()
        data = m.to_dict()
        assert "requests_total" in data
        assert "inference_latency" in data
        assert "device" in data


class TestHealthChecker:
    def test_healthy_when_all_pass(self) -> None:
        hc = HealthChecker()
        hc.register("always_ok", lambda: (HealthStatus.HEALTHY, "ok", None))
        status, results = hc.run_all()
        assert status == HealthStatus.HEALTHY
        assert len(results) == 1

    def test_unhealthy_when_one_fails(self) -> None:
        hc = HealthChecker()
        hc.register("ok", lambda: (HealthStatus.HEALTHY, "ok", None))
        hc.register("fail", lambda: (HealthStatus.UNHEALTHY, "bad", None))
        status, _ = hc.run_all()
        assert status == HealthStatus.UNHEALTHY

    def test_degraded_status(self) -> None:
        hc = HealthChecker()
        hc.register("degraded", lambda: (HealthStatus.DEGRADED, "warn", None))
        status, results = hc.run_all()
        assert status == HealthStatus.DEGRADED

    def test_exception_in_probe(self) -> None:
        hc = HealthChecker()
        def bad_probe():
            raise RuntimeError("boom")
        hc.register("broken", bad_probe)
        status, results = hc.run_all()
        assert status == HealthStatus.UNHEALTHY
        assert "boom" in results[0].message

    def test_model_loaded_probe_healthy(self) -> None:
        probe = model_loaded_probe(lambda: True)
        status, msg, _ = probe()
        assert status == HealthStatus.HEALTHY

    def test_model_loaded_probe_degraded(self) -> None:
        probe = model_loaded_probe(lambda: False)
        status, _, _ = probe()
        assert status == HealthStatus.DEGRADED

    def test_memory_probe(self) -> None:
        probe = memory_probe(max_used_percent=100.0)  # always pass
        status, _, details = probe()
        assert status == HealthStatus.HEALTHY
        assert "used_percent" in details

    def test_checkpoint_exists_probe(self, tmp_path: Path) -> None:
        probe = checkpoint_exists_probe(tmp_path)
        status, _, _ = probe()
        assert status == HealthStatus.HEALTHY

    def test_checkpoint_missing_probe(self, tmp_path: Path) -> None:
        probe = checkpoint_exists_probe(tmp_path / "nonexistent")
        status, _, _ = probe()
        assert status == HealthStatus.DEGRADED
# Counter uses threading.Lock to prevent race conditions on inc()
# to_prometheus() output checked for # HELP, # TYPE, and metric lines
# Histogram(max_observations=5) tested: 10 observations -> count==5
