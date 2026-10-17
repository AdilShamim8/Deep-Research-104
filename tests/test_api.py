"""
Integration tests for the FastAPI app via TestClient.

Hits the system endpoints (health, version, ready, providers) and
verifies response shapes. Does NOT call any LLM provider — those are
mocked in the model-specific tests.

Marked as @pytest.mark.unit so they run in the default CI filter.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient once for the module. TestClient runs the
    lifespan events (setup_logging + cache init) which we don't need
    for these unit-ish tests, so we keep the scope module-level."""
    app = create_app()
    return TestClient(app)


# ── Root ────────────────────────────────────────────────────────────────────


class TestRoot:
    def test_root_returns_endpoints(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["name"]
        assert "endpoints" in body
        assert "/api/v1/research" in body["endpoints"]
        assert "/health" in body["endpoints"]


# ── Health ────────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["python_version"]
        assert body["version"]
        assert body["build_date"] == "2026-10-17"
        assert "redis" in body
        assert "uptime_seconds" in body
        assert "timestamp" in body


# ── Version ──────────────────────────────────────────────────────────────────


class TestVersion:
    def test_version_returns_full_info(self, client: TestClient) -> None:
        r = client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "deep-research-104"
        assert body["version"] == "1.0.0"
        assert body["build_date"] == "2026-10-17"
        assert body["python_version"]
        assert body["app_env"]


# ── Readiness ──────────────────────────────────────────────────────────────────


class TestReady:
    def test_ready_returns_200_or_503(self, client: TestClient) -> None:
        r = client.get("/ready")
        assert r.status_code in (200, 503)
        body = r.json()
        assert body["status"] in ("ready", "not_ready")
        assert "checks" in body
        assert "timestamp" in body


# ── Providers ──────────────────────────────────────────────────────────────────


class TestProviders:
    def test_providers_returns_list(self, client: TestClient) -> None:
        r = client.get("/api/v1/providers")
        assert r.status_code == 200
        body = r.json()
        assert "providers" in body
        for p in ["openai", "deepseek", "ollama", "vllm", "llamacpp"]:
            assert p in body["providers"]


# ── Metrics ──────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_endpoint_exists(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        # Prometheus format: at least one line starting with '#'
        assert "#" in r.text

    def test_cache_metrics_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/v1/metrics/cache")
        assert r.status_code == 200
        body = r.json()
        for key in ("hits", "misses", "errors", "l1_hits", "l2_hits", "l3_hits", "hit_rate"):
            assert key in body

    def test_search_metrics_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/v1/metrics/search")
        assert r.status_code == 200
        body = r.json()
        # Returns dict keyed by provider name
        assert isinstance(body, dict)


# ── Route-level health ──────────────────────────────────────────────────────


class TestRouteHealth:
    def test_api_v1_health(self, client: TestClient) -> None:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


# ── Request validation ──────────────────────────────────────────────────────


class TestRequestValidation:
    def test_research_missing_question_returns_422(self, client: TestClient) -> None:
        r = client.post("/api/v1/research", json={})
        assert r.status_code == 422

    def test_research_question_too_short_returns_422(self, client: TestClient) -> None:
        r = client.post("/api/v1/research", json={"question": "ab"})
        assert r.status_code == 422

    def test_research_unknown_field_returns_422(self, client: TestClient) -> None:
        # extra='forbid' should reject unknown fields
        r = client.post(
            "/api/v1/research",
            json={"question": "what is X", "surprise_field": "boom"},
        )
        assert r.status_code == 422

    def test_research_unknown_provider_returns_422(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/research",
            json={"question": "what is X", "provider": "anthropic"},
        )
        assert r.status_code == 422

    def test_cot_with_invalid_n_samples(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/cot",
            json={"question": "x", "n_samples": 0},
        )
        assert r.status_code == 422


# ── Headers ─────────────────────────────────────────────────────────────────


class TestSecurityHeaders:
    def test_security_headers_present(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_request_id_header_in_response(self, client: TestClient) -> None:
        r = client.get("/health")
        # Our middleware stamps an X-Request-ID on every response
        assert "X-Request-ID" in r.headers
