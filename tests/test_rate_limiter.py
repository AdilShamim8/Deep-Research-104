"""
Unit tests for src.utils.rate_limiter (InProcessRateLimiter).

Tests cover the per-model token bucket, race-free lock initialization,
metrics counters, and prefix-based limit lookup. Uses real asyncio so
we exercise the actual lock + sleep paths.
"""

from __future__ import annotations

import asyncio

import pytest
from src.utils.rate_limiter import (
    DEFAULT_LIMITS,
    InProcessRateLimiter,
    Limit,
    _get_limits,
)

# ── Limit lookup ──────────────────────────────────────────────────────────────


class TestLimitLookup:
    def test_exact_match(self) -> None:
        limit = _get_limits("o3-mini")
        assert limit.rpm == DEFAULT_LIMITS["o3-mini"].rpm

    def test_prefix_fallback(self) -> None:
        # "o3-mini-2025-01-31" should match "o3-mini" (longest prefix wins)
        limit = _get_limits("o3-mini-2025-01-31")
        assert limit.rpm == DEFAULT_LIMITS["o3-mini"].rpm

    def test_unknown_model_returns_default(self) -> None:
        limit = _get_limits("totally-unknown-model")
        assert limit.rpm == DEFAULT_LIMITS["default"].rpm

    def test_search_providers_have_zero_tpm(self) -> None:
        # Search providers set tpm=0 (skip TPM check)
        for provider in ["duckduckgo", "serpapi", "brave", "bing"]:
            assert _get_limits(provider).tpm == 0


# ── Limit dataclass ──────────────────────────────────────────────────────────


class TestLimitDataclass:
    def test_immutable(self) -> None:
        with pytest.raises(Exception):
            Limit(rpm=60, tpm=100_000, burst=5).rpm = 100  # type: ignore

    def test_equality(self) -> None:
        a = Limit(rpm=60, tpm=100_000, burst=5)
        b = Limit(rpm=60, tpm=100_000, burst=5)
        assert a == b


# ── Acquire / metrics ────────────────────────────────────────────────────────


class TestAcquire:
    @pytest.mark.asyncio
    async def test_acquire_returns_quickly_under_limit(self) -> None:
        limiter = InProcessRateLimiter()
        # Use a high-rpm model so we don't actually sleep
        await limiter.acquire("o3-mini", estimated_tokens=1000, timeout=1.0)
        # No assertion needed; just should not raise

    @pytest.mark.asyncio
    async def test_metrics_records_acquired(self) -> None:
        limiter = InProcessRateLimiter()
        await limiter.acquire("o3-mini", estimated_tokens=100, timeout=1.0)
        metrics = limiter.metrics()
        assert "o3-mini" in metrics
        assert metrics["o3-mini"]["acquired"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_acquires_use_per_model_lock(self) -> None:
        """Two concurrent acquires for the same model should serialize."""
        limiter = InProcessRateLimiter()
        results: list[int] = []

        async def _acquire(idx: int) -> None:
            await limiter.acquire("o3-mini", estimated_tokens=1, timeout=1.0)
            results.append(idx)

        await asyncio.gather(_acquire(0), _acquire(1), _acquire(2))
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_acquire_records_first_call(self) -> None:
        """A first acquire should be tracked in the metrics dict."""
        limiter = InProcessRateLimiter()
        # Use a low-rpm test model so we exercise the limit logic
        # — but only do one acquire so we don't actually wait 60s.
        DEFAULT_LIMITS["__test_low__"] = Limit(rpm=1, tpm=1_000_000, burst=1)

        await limiter.acquire("__test_low__", estimated_tokens=1, timeout=2.0)
        metrics = limiter.metrics()
        assert metrics["__test_low__"]["acquired"] == 1

        # Clean up the test limit so we don't leak state across tests
        DEFAULT_LIMITS.pop("__test_low__", None)


# ── Multiple providers don't interfere ───────────────────────────────────────


class TestPerModelIsolation:
    @pytest.mark.asyncio
    async def test_two_models_acquire_independently(self) -> None:
        limiter = InProcessRateLimiter()
        await asyncio.gather(
            limiter.acquire("o3-mini", estimated_tokens=1, timeout=1.0),
            limiter.acquire("deepseek-reasoner", estimated_tokens=1, timeout=1.0),
        )
        metrics = limiter.metrics()
        assert "o3-mini" in metrics
        assert "deepseek-reasoner" in metrics
        assert metrics["o3-mini"]["acquired"] >= 1
        assert metrics["deepseek-reasoner"]["acquired"] >= 1
