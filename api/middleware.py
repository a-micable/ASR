"""FastAPI middleware: logging, errors, rate limiting, performance."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log incoming requests and responses with timing."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info(
            "Request started",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else "unknown",
                }
            },
        )

        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Request completed",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            },
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(duration_ms, 2))
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global exception handler middleware."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await call_next(request)
        except ValueError as exc:
            logger.warning("Validation error: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"detail": str(exc), "type": "validation_error"},
            )
        except FileNotFoundError as exc:
            logger.error("Resource not found: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"detail": str(exc), "type": "not_found"},
            )
        except Exception as exc:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.exception("Unhandled error [%s]: %s", request_id, exc)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error",
                    "type": "server_error",
                    "request_id": request_id,
                },
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.

    For production, prefer Redis-backed limiter (e.g. slowapi with Redis).
    """

    def __init__(
        self,
        app: FastAPI,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_key(self, request: Request) -> str:
        if request.client:
            return request.client.host
        return "unknown"

    def _is_rate_limited(self, client_key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self._requests[client_key] = [
            t for t in self._requests[client_key] if t > window_start
        ]
        if len(self._requests[client_key]) >= self.max_requests:
            return True
        self._requests[client_key].append(now)
        return False

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path in {"/health", "/metrics", "/docs", "/openapi.json"}:
            return await call_next(request)

        client_key = self._get_client_key(request)
        if self._is_rate_limited(client_key):
            logger.warning("Rate limit exceeded for %s", client_key)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": str(self.window_seconds)},
            )
        return await call_next(request)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Track API performance metrics in application state."""

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        metrics = getattr(request.app.state, "metrics", None)
        if metrics is not None:
            metrics["total_requests"] = metrics.get("total_requests", 0) + 1
            if response.status_code >= 400:
                metrics["error_count"] = metrics.get("error_count", 0) + 1
            if request.url.path == "/transcribe":
                metrics["transcription_count"] = metrics.get("transcription_count", 0) + 1
                latencies = metrics.setdefault("latencies_ms", [])
                latencies.append(duration_ms)
                if len(latencies) > 1000:
                    metrics["latencies_ms"] = latencies[-1000:]

        return response


def register_middleware(app: FastAPI, rate_limit: str = "60/minute") -> None:
    """
    Register all middleware on the FastAPI app.

    Args:
        app: FastAPI application.
        rate_limit: Rate limit string, e.g. '60/minute'.
    """
    parts = rate_limit.split("/")
    max_requests = int(parts[0]) if parts else 60
    window = 60
    if len(parts) > 1 and "minute" in parts[1]:
        window = 60
    elif len(parts) > 1 and "second" in parts[1]:
        window = 1

    app.add_middleware(PerformanceMonitoringMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=max_requests, window_seconds=window)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )
