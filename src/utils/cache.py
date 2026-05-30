"""
Multi-layer caching: in-memory (fast) + Redis (persistent).
Used to cache search results and LLM responses.
"""

import json
import hashlib
import asyncio
from typing import Any, Optional
from datetime import timedelta
import diskcache
import aioredis
from loguru import logger
from config.settings import settings


def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Create deterministic cache key from arguments."""
    key_data = json.dumps(
        {"args": list(args), "kwargs": kwargs},
        sort_keys=True,
        default=str
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


class LayeredCache:
    """
    Two-layer cache:
    L1: In-process disk cache (diskcache) - fast, local
    L2: Redis - shared across workers, persistent
    """
    
    def __init__(
        self,
        disk_cache_dir: str = ".cache",
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
    ):
        self.default_ttl = default_ttl
        self._disk_cache = diskcache.Cache(disk_cache_dir)
        self._redis_url = redis_url or settings.redis_url
        self._redis: Optional[aioredis.Redis] = None
    
    async def _get_redis(self) -> Optional[aioredis.Redis]:
        """Lazy Redis connection with graceful fallback."""
        if self._redis is not None:
            return self._redis
        try:
            self._redis = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._redis.ping()
            logger.info("Redis cache connected")
            return self._redis
        except Exception as e:
            logger.warning(f"Redis unavailable, using disk cache only: {e}")
            return None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache, checking L1 then L2."""
        # L1: Disk cache
        if key in self._disk_cache:
            logger.debug(f"Cache L1 hit: {key[:16]}...")
            return self._disk_cache[key]
        
        # L2: Redis
        redis = await self._get_redis()
        if redis:
            try:
                value = await redis.get(key)
                if value:
                    logger.debug(f"Cache L2 hit: {key[:16]}...")
                    parsed = json.loads(value)
                    # Populate L1
                    self._disk_cache.set(key, parsed, expire=self.default_ttl)
                    return parsed
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set in both cache layers."""
        ttl = ttl or self.default_ttl
        
        # L1: Disk cache
        try:
            self._disk_cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.warning(f"Disk cache set error: {e}")
        
        # L2: Redis
        redis = await self._get_redis()
        if redis:
            try:
                serialized = json.dumps(value, default=str)
                await redis.setex(key, ttl, serialized)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
    
    async def invalidate(self, key: str) -> None:
        """Remove from both layers."""
        self._disk_cache.delete(key)
        redis = await self._get_redis()
        if redis:
            await redis.delete(key)
    
    async def close(self) -> None:
        """Clean up connections."""
        if self._redis:
            await self._redis.close()
        self._disk_cache.close()


# Global cache instance
cache = LayeredCache()