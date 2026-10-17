"""
Unit tests for src.utils.cache (LayeredCache + LRU + helpers).

No network — Redis is never actually connected (we monkeypatch _get_redis
to return None). Tests focus on the LRU layer, key derivation, TTL
handling, and graceful degradation when Redis is down.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from src.utils.cache import (
    LayeredCache,
    LRUCache,
    TTL,
    make_cache_key,
    make_namespace_key,
)


# ── Key derivation ────────────────────────────────────────────────────────────

class TestKeyDerivation:
    def test_make_cache_key_deterministic(self) -> None:
        k1 = make_cache_key("a", "b", x=1)
        k2 = make_cache_key("a", "b", x=1)
        assert k1 == k2

    def test_make_cache_key_changes_with_args(self) -> None:
        assert make_cache_key("a") != make_cache_key("b")

    def test_make_cache_key_changes_with_kwargs(self) -> None:
        assert make_cache_key("a", x=1) != make_cache_key("a", x=2)

    def test_make_cache_key_kwargs_order_irrelevant(self) -> None:
        assert make_cache_key("a", x=1, y=2) == make_cache_key("a", y=2, x=1)

    def test_make_cache_key_is_sha256(self) -> None:
        k = make_cache_key("test")
        assert len(k) == 64  # 32 bytes hex
        assert all(c in "0123456789abcdef" for c in k)

    def test_make_namespace_key_has_prefix(self) -> None:
        k = make_namespace_key("search", "q")
        assert k.startswith("search:")

    def test_different_namespaces_differ(self) -> None:
        assert make_namespace_key("search", "q") != make_namespace_key("llm", "q")


# ── TTL presets ──────────────────────────────────────────────────────────────

class TestTTLPresets:
    def test_short_is_60(self) -> None:
        assert TTL.SHORT == 60

    def test_medium_is_3600(self) -> None:
        assert TTL.MEDIUM == 3_600

    def test_long_is_86400(self) -> None:
        assert TTL.LONG == 86_400

    def test_week_is_604800(self) -> None:
        assert TTL.WEEK == 604_800

    def test_search_results_is_hour(self) -> None:
        assert TTL.SEARCH_RESULTS == 3_600


# ── In-process LRU ───────────────────────────────────────────────────────────

class TestLRUCache:
    @pytest.mark.asyncio
    async def test_set_then_get(self) -> None:
        lru = LRUCache(capacity=4)
        await lru.set("k", "v")
        assert await lru.get("k") == "v"

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self) -> None:
        lru = LRUCache()
        assert await lru.get("missing") is None

    @pytest.mark.asyncio
    async def test_capacity_eviction_oldest_first(self) -> None:
        lru = LRUCache(capacity=2)
        await lru.set("a", 1)
        await lru.set("b", 2)
        await lru.set("c", 3)  # evicts "a"
        assert await lru.get("a") is None
        assert await lru.get("b") == 2
        assert await lru.get("c") == 3

    @pytest.mark.asyncio
    async def test_get_promotes_to_most_recent(self) -> None:
        lru = LRUCache(capacity=2)
        await lru.set("a", 1)
        await lru.set("b", 2)
        await lru.get("a")  # touch "a", it should now be most-recent
        await lru.set("c", 3)  # should evict "b" instead of "a"
        assert await lru.get("a") == 1
        assert await lru.get("b") is None

    @pytest.mark.asyncio
    async def test_delete_removes_key(self) -> None:
        lru = LRUCache()
        await lru.set("k", "v")
        await lru.delete("k")
        assert await lru.get("k") is None

    @pytest.mark.asyncio
    async def test_clear_wipes_all(self) -> None:
        lru = LRUCache()
        await lru.set("a", 1)
        await lru.set("b", 2)
        await lru.clear()
        assert await lru.get("a") is None
        assert await lru.get("b") is None

    @pytest.mark.asyncio
    async def test_ttl_expires(self) -> None:
        lru = LRUCache()
        await lru.set("k", "v", ttl=1)
        assert await lru.get("k") == "v"
        await asyncio.sleep(1.1)
        assert await lru.get("k") is None


# ── Layered cache with Redis disabled ──────────────────────────────────────────

class TestLayeredCacheNoRedis:
    """Run with Redis explicitly disabled so we test the disk + LRU fallback."""

    def setup_method(self) -> None:
        # Fresh cache per test; disk cache disabled too (we only test LRU)
        self.cache = LayeredCache(default_ttl=60, lru_capacity=8)
        # Stub out disk cache so we don't write to disk during tests
        self.cache._get_disk_cache = lambda: None  # type: ignore
        # Stub out Redis
        async def _no_redis() -> None:
            return None
        self.cache._get_redis = _no_redis  # type: ignore

    @pytest.mark.asyncio
    async def test_set_then_get_via_lru(self) -> None:
        await self.cache.set("k", {"a": 1})
        result = await self.cache.get("k")
        assert result == {"a": 1}

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self) -> None:
        assert await self.cache.get("missing") is None

    @pytest.mark.asyncio
    async def test_miss_increments_metric(self) -> None:
        before = self.cache.metrics.misses
        await self.cache.get("absent")
        assert self.cache.metrics.misses == before + 1

    @pytest.mark.asyncio
    async def test_hit_increments_metric(self) -> None:
        await self.cache.set("k", "v")
        await self.cache.get("k")
        assert self.cache.metrics.hits >= 1

    @pytest.mark.asyncio
    async def test_invalidate_removes_key(self) -> None:
        await self.cache.set("k", "v")
        await self.cache.invalidate("k")
        assert await self.cache.get("k") is None

    @pytest.mark.asyncio
    async def test_set_many_bulk(self) -> None:
        await self.cache.set_many({"a": 1, "b": 2, "c": 3}, ttl=60)
        assert await self.cache.get("a") == 1
        assert await self.cache.get("b") == 2
        assert await self.cache.get("c") == 3

    @pytest.mark.asyncio
    async def test_get_many_only_returns_hits(self) -> None:
        await self.cache.set("a", 1)
        await self.cache.set("b", 2)
        result = await self.cache.get_many(["a", "b", "missing"])
        assert "a" in result and "b" in result
        assert "missing" not in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        await self.cache.close()
        await self.cache.close()  # should not raise
