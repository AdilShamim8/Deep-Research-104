"""
End-to-end smoke test: verify the FastAPI app boots and serves the
system endpoints without crashing.

Run:
    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Spin up the full FastAPI app once for the whole module."""
    from src.api.main import create_app
    app = create_app()
    return TestClient(app)


class TestAppBoots:
    """Sanity checks — if any of these fail, the entire deployment is broken."""

    def test_app_object_exists(self, client: TestClient) -> None:
        assert client is not None

    def test_root_responds(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200

    def test_health_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_ready_returns_response(self, client: TestClient) -> None:
        r = client.get("/ready")
        assert r.status_code in (200, 503)

    def test_version_returns_response(self, client: TestClient) -> None:
        r = client.get("/version")
        assert r.status_code == 200

    def test_metrics_returns_prometheus_text(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "deep_research_" in r.text or "#" in r.text

    def test_providers_lists_known(self, client: TestClient) -> None:
        r = client.get("/api/v1/providers")
        assert r.status_code == 200
        assert "openai" in r.json()["providers"]


class TestAppResilience:
    """Negative-path checks — the app should reject malformed requests cleanly."""

    def test_unknown_route_returns_404(self, client: TestClient) -> None:
        r = client.get("/api/v1/does-not-exist")
        assert r.status_code == 404

    def test_post_to_get_only_route_returns_405(self, client: TestClient) -> None:
        r = client.post("/health")
        assert r.status_code == 405

    def test_invalid_json_returns_422(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/research",
            data="not-json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 422

    def test_huge_body_rejected_413(self, client: TestClient) -> None:
        # Body size cap is 1 MiB by default
        huge = "x" * (2 * 1024 * 1024)
        r = client.post(
            "/api/v1/research",
            json={"question": huge},
        )
        # Either 413 (body cap) or 422 (field max_length), both are acceptable
        assert r.status_code in (413, 422)
