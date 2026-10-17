"""
Shared pytest fixtures + config for the Deep Research 104 test suite.

Sets:
    APP_ENV=test, REDIS_URL=""  → tests use in-memory fallbacks
    pytest-asyncio in auto mode (already configured in pyproject.toml)

Provides:
    - `mock_model` — a stub BaseModel returning canned responses
    - `sample_search_response` — a SearchResponse with 2 results
    - `tmp_cache` — fresh LayeredCache per test (Redis + disk stubbed)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import pytest

# ── Test env ────────────────────────────────────────────────────────────────
# Force these before any project import happens.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("REDIS_URL", "")  # disable Redis in tests
os.environ.setdefault("OPENAI_API_KEY", "")  # tests don't call OpenAI
os.environ.setdefault("DEEPSEEK_API_KEY", "")


# ── pytest markers (mirror pyproject.toml) ───────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests (no network)")
    config.addinivalue_line("markers", "integration: requires Redis/Ollama/OpenAI")
    config.addinivalue_line("markers", "slow: takes more than 5s")


# ── Stub BaseModel ──────────────────────────────────────────────────────────

from src.models.base_model import (
    BaseModel,
    GenerationConfig,
    ModelCapability,
    ModelResponse,
)


class StubModel(BaseModel):
    """Deterministic stub model for tests. Returns a canned response."""

    def __init__(self, model_name: str = "stub-1", canned_content: str = "stub answer"):
        super().__init__(model_name)
        self.canned_content = canned_content
        self.capabilities = {ModelCapability.STREAMING, ModelCapability.REASONING}
        self.call_count = 0

    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        self.call_count += 1
        # Approximate token usage so cost tracking tests pass
        text = " ".join(m.get("content", "") for m in messages)
        return ModelResponse(
            content=self.canned_content,
            reasoning_content=None,
            model=self.model_name,
            prompt_tokens=max(1, len(text) // 4),
            completion_tokens=max(1, len(self.canned_content) // 4),
            total_tokens=max(2, (len(text) + len(self.canned_content)) // 4),
            latency_ms=10.0,
        )

    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        # Yield a few tokens from the canned content
        for word in self.canned_content.split():
            yield word + " "

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_context_limit(self) -> int:
        return 16_384


@pytest.fixture
def mock_model() -> StubModel:
    return StubModel()


# ── Sample search response ──────────────────────────────────────────────────

from src.search.web_search import SearchResult, SearchResponse


@pytest.fixture
def sample_search_response() -> SearchResponse:
    return SearchResponse(
        query="test query",
        results=[
            SearchResult(
                title="Result 1",
                url="https://example.com/page1",
                snippet="This is the first result.",
                position=1,
                provider="duckduckgo",
            ),
            SearchResult(
                title="Result 2",
                url="https://example.com/page2",
                snippet="This is the second result.",
                position=2,
                provider="duckduckgo",
            ),
        ],
        provider="duckduckgo",
        total_results=2,
        search_time_ms=42.0,
    )


# ── Fresh cache per test ────────────────────────────────────────────────────

from src.utils.cache import LayeredCache


@pytest.fixture
async def tmp_cache() -> LayeredCache:
    cache = LayeredCache(default_ttl=60, lru_capacity=8)
    # Stub disk + Redis so tests are hermetic
    cache._get_disk_cache = lambda: None  # type: ignore
    async def _no_redis() -> None:
        return None
    cache._get_redis = _no_redis  # type: ignore
    yield cache
    await cache.close()
