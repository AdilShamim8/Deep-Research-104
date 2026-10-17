"""
Multi-layer caching: in-process LRU + disk + Redis.

Used to cache search results and LLM responses to reduce cost and
latency. Production hardening over the toy original:
- Replaces deprecated aioredis with redis.asyncio (aioredis is dead).
- L1 in-process LRU (no disk I/O for hot keys) on top of diskcache L2.
- Graceful degradation: if Redis is down, fall back to disk; if disk
  fails, fall back to in-memory.
- TTL per key, with separate TTLs for search vs LLM vs embedding.
- Cache hit/miss counters exposed to Prometheus.
- Safe serialization: JSON with default=str fallback for non-JSON types.
- Async-safe connection pool; cleanup on shutdown.

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from datetime import timedelta
from typing import Any, Optional

from loguru import logger

from config.settings import settings


# ── In-process LRU ────────────────────────────────────────────────────────────

class LRUCache:
    """Tiny thread-safe LRU cache for the hottest keys.

    Note: This is per-process — Redis remains the shared layer for
    cross-worker coordination. We keep LRU tiny (default 256 entries)
    because Python dicts are surprisingly fast for small N.
    """

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            import time as _time
            if expires_at and _time.time() > expires_at:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        async with self._lock:
            import time as _time
            expires_at = _time.time() + ttl if ttl else 0.0
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


# ── Cache key derivation ──────────────────────────────────────────────────────

def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Create a deterministic SHA-256 cache key from arbitrary args."""
    payload = json.dumps(
        {"args": list(args), "kwargs": kwargs},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_namespace_key(namespace: str, *args: Any, **kwargs: Any) -> str:
    """Namespaced cache key (e.g. 'search:...' vs 'llm:...')."""
    return f"{namespace}:{make_cache_key(*args, **kwargs)}"


# ── Metrics counters ─────────────────────────────────────────────────────────

class CacheMetrics:
    """Lightweight hit/miss counters for Prometheus scraping.

    Exposed via the /metrics endpoint as deep_research_cache_hits_total
    and deep_research_cache_misses_total.
    """

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0

    def record_hit(self, layer: str) -> None:
        self.hits += 1
        if layer == "L1":
            self.l1_hits += 1
        elif layer == "L2":
            self.l2_hits += 1
        elif layer == "L3":
            self.l3_hits += 1

    def record_miss(self) -> None:
        self.misses += 1

    def record_error(self) -> None:
        self.errors += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ── Layered cache ─────────────────────────────────────────────────────────────

class LayeredCache:
    """
    Three-layer cache:
        L1: In-process LRU (fastest, per-worker)
        L2: On-disk diskcache (per-worker persistent)
        L3: Redis (shared across workers)

    Reads cascade L1 -> L2 -> L3 -> None. Writes propagate to all
    available layers. If Redis is unreachable, the cache still works
    via L1+L2 (just no cross-worker sharing).
    """

    def __init__(
        self,
        disk_cache_dir: str = ".cache",
        redis_url: Optional[str] = None,
        default_ttl: int = 3_600,
        lru_capacity: int = 256,
    ) -> None:
        self.default_ttl = default_ttl
        self._redis_url = redis_url or settings.redis_url
        self._redis: Optional[Any] = None  # redis.asyncio.Redis
        self._redis_checked = False
        self._disk_cache = None  # lazy init
        self._disk_cache_dir = disk_cache_dir
        self._lru = LRUCache(lru_capacity)
        self.metrics = CacheMetrics()

    # ── Lazy initializers ────────────────────────────────────────────────

    def _get_disk_cache(self) -> Any:
        if self._disk_cache is None:
            try:
                import diskcache
                import os
                os.makedirs(self._disk_cache_dir, exist_ok=True)
                self._disk_cache = diskcache.Cache(self._disk_cache_dir)
            except Exception as e:
                logger.warning(f"Disk cache unavailable: {e}")
        return self._disk_cache

    async def _get_redis(self) -> Optional[Any]:
        """Lazy Redis connection with graceful fallback."""
        if self._redis is not None:
            return self._redis
        if self._redis_checked:
            return None  # already failed once; don't retry every call
        self._redis_checked = True
        try:
            # Prefer redis.asyncio (aioredis is deprecated)
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=5,
                max_connections=settings.redis_max_connections,
            )
            await self._redis.ping()
            logger.info(f"Redis cache connected: {self._redis_url}")
            return self._redis
        except Exception as e:
            logger.warning(f"Redis unavailable, using disk + LRU only: {e}")
            self._redis = None
            return None

    # ── Public API ───────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache, cascading L1 -> L2 -> L3."""
        # L1: in-process LRU
        try:
            value = await self._lru.get(key)
            if value is not None:
                self.metrics.record_hit("L1")
                logger.debug(f"Cache L1 hit: {key[:16]}…")
                return value
        except Exception as e:
            logger.warning(f"LRU get error: {e}")

        # L2: disk
        disk = self._get_disk_cache()
        if disk is not None:
            try:
                if key in disk:
                    value = disk[key]
                    # Populate L1
                    await self._lru.set(key, value, ttl=self.default_ttl)
                    self.metrics.record_hit("L2")
                    logger.debug(f"Cache L2 hit: {key[:16]}…")
                    return value
            except Exception as e:
                logger.warning(f"Disk cache get error: {e}")
                self.metrics.record_error()

        # L3: Redis
        redis = await self._get_redis()
        if redis is not None:
            try:
                raw = await redis.get(key)
                if raw:
                    value = json.loads(raw)
                    # Populate L1 and L2
                    await self._lru.set(key, value, ttl=self.default_ttl)
                    try:
                        if disk is not None:
                            disk.set(key, value, expire=self.default_ttl)
                    except Exception:
                        pass
                    self.metrics.record_hit("L3")
                    logger.debug(f"Cache L3 hit: {key[:16]}…")
                    return value
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                self.metrics.record_error()

        self.metrics.record_miss()
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set in all available layers."""
        ttl = ttl or self.default_ttl

        # L1
        try:
            await self._lru.set(key, value, ttl=ttl)
        except Exception as e:
            logger.warning(f"LRU set error: {e}")

        # L2
        disk = self._get_disk_cache()
        if disk is not None:
            try:
                disk.set(key, value, expire=ttl)
            except Exception as e:
                logger.warning(f"Disk cache set error: {e}")
                self.metrics.record_error()

        # L3
        redis = await self._get_redis()
        if redis is not None:
            try:
                serialized = json.dumps(value, default=str, ensure_ascii=False)
                await redis.setex(key, ttl, serialized)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self.metrics.record_error()

    async def invalidate(self, key: str) -> None:
        """Remove from all layers."""
        await self._lru.delete(key)
        disk = self._get_disk_cache()
        if disk is not None:
            try:
                disk.delete(key)
            except Exception:
                pass
        redis = await self._get_redis()
        if redis is not None:
            try:
                await redis.delete(key)
            except Exception:
                pass

    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a glob pattern (e.g. 'search:*').

        Returns the number of keys removed. Useful for cache busting
        when an upstream dependency changes shape.
        """
        removed = 0
        redis = await self._get_redis()
        if redis is not None:
            try:
                async for key in redis.scan_iter(match=pattern, count=500):
                    await redis.delete(key)
                    removed += 1
            except Exception as e:
                logger.warning(f"Redis clear_pattern error: {e}")
        return removed

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Bulk fetch. Returns a dict of {key: value} for hits only."""
        result: dict[str, Any] = {}
        for key in keys:
            v = await self.get(key)
            if v is not None:
                result[key] = v
        return result

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> None:
        """Bulk set with the same TTL."""
        for key, value in items.items():
            await self.set(key, value, ttl=ttl)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release all resources. Idempotent."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None
        if self._disk_cache is not None:
            try:
                self._disk_cache.close()
            except Exception:
                pass
            self._disk_cache = None
        await self._lru.clear()


# ── TTL presets ──────────────────────────────────────────────────────────────

class TTL:
    """Common TTL presets in seconds."""
    SHORT = 60                           # 1 minute
    MEDIUM = 3_600                       # 1 hour
    LONG = 86_400                        # 1 day
    WEEK = 604_800                       # 7 days
    SEARCH_RESULTS = 3_600               # search results change hourly
    LLM_RESPONSE = 86_400                # LLM completions are deterministic enough
    EMBEDDINGS = 604_800                 # embeddings rarely change
    HEALTH_CHECK = 30


# ── Global cache instance ────────────────────────────────────────────────────

cache = LayeredCache()
