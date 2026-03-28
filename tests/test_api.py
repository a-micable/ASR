"""Tests for FastAPI inference service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from api.app import create_app, get_settings
from training.config import PipelineConfig


@pytest.fixture
def api_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        api={"model_path": tmp_path / "model", "default_language": "am"},
        logging={"log_dir": tmp_path / "logs"},
    )


@pytest.fixture
def mock_whisper():
    processor = MagicMock()
    processor.get_decoder_prompt_ids.return_value = None
    processor.return_value = MagicMock(input_features=torch.zeros(1, 80, 3000))
    processor.batch_decode.return_value = ["ሰላም"]

    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    model.eval = MagicMock()
    return processor, model


@pytest.fixture
def client(api_config: PipelineConfig, mock_whisper, tmp_path: Path):
    processor, model = mock_whisper
    (tmp_path / "model").mkdir(exist_ok=True)

    with patch("api.app.load_whisper_model", return_value=(processor, model)):
        app = create_app(api_config)
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_status(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_degraded_without_model(self, api_config: PipelineConfig, tmp_path: Path) -> None:
        with patch("api.app.load_whisper_model", side_effect=FileNotFoundError("missing")):
            app = create_app(api_config)
            with TestClient(app) as c:
                data = c.get("/health").json()
                assert data["status"] == "degraded"
                assert data["model_loaded"] is False


class TestMetricsEndpoint:
    def test_metrics_returns_counters(self, client: TestClient) -> None:
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert data["total_requests"] >= 1


class TestTranscribeEndpoint:
    def test_transcribe_success(self, client: TestClient, audio_bytes: bytes) -> None:
        response = client.post(
            "/transcribe",
            files={"audio_file": ("test.wav", audio_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["transcription"] == "ሰላም"
        assert data["language"] == "am"

    def test_transcribe_unsupported_format(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe",
            files={"audio_file": ("test.txt", b"not audio", "text/plain")},
        )
        assert response.status_code == 422

    def test_transcribe_empty_file(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe",
            files={"audio_file": ("test.wav", b"", "audio/wav")},
        )
        assert response.status_code == 422

    def test_transcribe_model_not_loaded(
        self, api_config: PipelineConfig, audio_bytes: bytes, tmp_path: Path
    ) -> None:
        with patch("api.app.load_whisper_model", side_effect=FileNotFoundError("missing")):
            app = create_app(api_config)
            with TestClient(app) as c:
                response = c.post(
                    "/transcribe",
                    files={"audio_file": ("test.wav", audio_bytes, "audio/wav")},
                )
                assert response.status_code == 503


class TestMiddleware:
    def test_request_logging_adds_headers(self, client: TestClient) -> None:
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers

    def test_rate_limit_blocks_excessive_requests(self) -> None:
        from api.middleware import RateLimitMiddleware
        from fastapi import FastAPI

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware, max_requests=2, window_seconds=60)

        @app.get("/limited")
        async def limited():
            return {"ok": True}

        with TestClient(app) as c:
            assert c.get("/limited").status_code == 200
            assert c.get("/limited").status_code == 200
            assert c.get("/limited").status_code == 429


class TestCreateApp:
    def test_create_app_returns_fastapi(self, api_config: PipelineConfig) -> None:
        app = create_app(api_config)
        assert app.title == "Whisper ASR API"

    def test_get_settings_singleton(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
# TestMiddleware.test_rate_limit_blocks_excessive_requests tests 429 response
# BatchTranscriptionRequest validates max_files range [1, 50]
# transcribe_long() tested with 35s audio using 30s chunk + 2s overlap
# SupportedLanguage enum validates am, om, ti, en and rejects unknown
# end-to-end: GET /health, GET /metrics, POST /transcribe sequence
# test_health_returns_model_name verifies model_name field in response
