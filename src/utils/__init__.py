"""
Cross-cutting utilities for Deep Research 104.

Public surface:
    - TokenCounter, count_tokens     — context-window-safe token counting
    - rate_limiter                     — per-model RPM/TPM token bucket
    - cache, make_cache_key,
      make_namespace_key, TTL          — 3-layer LRU+disk+Redis cache

Date tag: 17 October 2026.
"""

from src.utils.cache import TTL, LayeredCache, cache, make_cache_key, make_namespace_key
from src.utils.rate_limiter import InProcessRateLimiter, Limit, rate_limiter
from src.utils.token_counter import TokenCounter, count_tokens

__all__ = [
    "TokenCounter",
    "count_tokens",
    "rate_limiter",
    "InProcessRateLimiter",
    "Limit",
    "cache",
    "LayeredCache",
    "make_cache_key",
    "make_namespace_key",
    "TTL",
]
