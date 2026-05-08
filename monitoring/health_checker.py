"""Health check probes for liveness and readiness endpoints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Possible health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"


@dataclass
class CheckResult:
    """Result of a single health probe."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: dict[str, Any] | None = None


class HealthChecker:
    """
    Composite health checker that runs multiple probes.

    Each probe is a callable returning (HealthStatus, str, dict | None).
    Probes run in registration order; the first UNHEALTHY probe short-circuits
    to an unhealthy aggregate.
    """

    def __init__(self) -> None:
        self._probes: list[tuple[str, Callable[[], tuple[HealthStatus, str, dict | None]]]] = []

    def register(
        self,
        name: str,
        probe: Callable[[], tuple[HealthStatus, str, dict | None]],
    ) -> None:
        """
        Register a health probe.

        Args:
            name: Human-readable probe name.
            probe: Zero-argument callable returning (status, message, details).
        """
        self._probes.append((name, probe))

    def run_all(self) -> tuple[HealthStatus, list[CheckResult]]:
        """
        Execute all registered probes.

        Returns:
            (aggregate_status, individual_results)
        """
        results: list[CheckResult] = []
        aggregate = HealthStatus.HEALTHY

        for name, probe in self._probes:
            start = time.perf_counter()
            try:
                status, message, details = probe()
            except Exception as exc:
                status = HealthStatus.UNHEALTHY
                message = f"Probe raised exception: {exc}"
                details = None
            duration_ms = (time.perf_counter() - start) * 1000
            results.append(CheckResult(name, status, message, round(duration_ms, 2), details))

            if status == HealthStatus.UNHEALTHY:
                aggregate = HealthStatus.UNHEALTHY
            elif status == HealthStatus.DEGRADED and aggregate == HealthStatus.HEALTHY:
                aggregate = HealthStatus.DEGRADED

        return aggregate, results


# ------------------------------------------------------------------
# Built-in probe factories
# ------------------------------------------------------------------

def model_loaded_probe(
    is_loaded: Callable[[], bool],
) -> Callable[[], tuple[HealthStatus, str, dict | None]]:
    """Return a probe that checks whether the model is loaded."""
    def _probe() -> tuple[HealthStatus, str, dict | None]:
        if is_loaded():
            return HealthStatus.HEALTHY, "Model is loaded", None
        return HealthStatus.DEGRADED, "Model is not loaded; service in degraded mode", None
    return _probe


def disk_space_probe(
    path: str | Path = "/",
    min_free_gb: float = 1.0,
) -> Callable[[], tuple[HealthStatus, str, dict | None]]:
    """Return a probe that checks available disk space."""
    def _probe() -> tuple[HealthStatus, str, dict | None]:
        usage = psutil.disk_usage(str(path))
        free_gb = usage.free / (1024**3)
        details = {
            "free_gb": round(free_gb, 2),
            "total_gb": round(usage.total / (1024**3), 2),
            "used_percent": usage.percent,
        }
        if free_gb < min_free_gb:
            return (
                HealthStatus.UNHEALTHY,
                f"Low disk space: {free_gb:.1f} GB free (minimum {min_free_gb} GB)",
                details,
            )
        return HealthStatus.HEALTHY, f"{free_gb:.1f} GB free", details
    return _probe


def memory_probe(
    max_used_percent: float = 90.0,
) -> Callable[[], tuple[HealthStatus, str, dict | None]]:
    """Return a probe that checks system memory utilization."""
    def _probe() -> tuple[HealthStatus, str, dict | None]:
        vm = psutil.virtual_memory()
        details = {
            "used_percent": vm.percent,
            "available_mb": round(vm.available / (1024**2), 1),
            "total_mb": round(vm.total / (1024**2), 1),
        }
        if vm.percent > max_used_percent:
            return (
                HealthStatus.DEGRADED,
                f"Memory pressure: {vm.percent:.0f}% used",
                details,
            )
        return HealthStatus.HEALTHY, f"{vm.percent:.0f}% used", details
    return _probe


def checkpoint_exists_probe(
    model_path: str | Path,
) -> Callable[[], tuple[HealthStatus, str, dict | None]]:
    """Return a probe that checks whether the model checkpoint directory exists."""
    def _probe() -> tuple[HealthStatus, str, dict | None]:
        p = Path(model_path)
        if p.exists():
            return HealthStatus.HEALTHY, f"Checkpoint found at {p}", {"path": str(p)}
        return HealthStatus.DEGRADED, f"Checkpoint not found at {p}", {"path": str(p)}
    return _probe
# readiness: model loaded; liveness: process running and memory available
# logger.warning used when memory usage > max_used_percent threshold
# disk_space_probe reports free_gb in details dict for alerting
# model_loaded probe checks app.state.model is not None
