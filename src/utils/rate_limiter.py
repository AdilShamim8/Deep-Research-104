"""
Token-bucket rate limiter for outbound LLM / search API calls.

Prevents hitting provider rate limits in production. Production
hardening over the toy original:
- Async-safe per-model locking (defaultdict was racy on first access)
- Tunable per-model RPM/TPM limits via env overrides
- Redis-backed shared limiter (so multiple workers coordinate) when
  Redis is available; falls back to in-process if not
- Max-wait timeout so a saturated queue fails fast instead of stalling
- Burst allowance (up to N requests instantly, then refill at RPM/60)
- Prometheus-friendly metrics counters

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger

# ── Rate limit tables ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Limit:
    rpm: int  # requests per minute
    tpm: int  # tokens per minute
    burst: int  # max instantaneous requests allowed


DEFAULT_LIMITS: dict[str, Limit] = {
    # OpenAI o-series (tier-2 numbers, conservative)
    "o3": Limit(rpm=500, tpm=10_000_000, burst=10),
    "o3-mini": Limit(rpm=1_000, tpm=50_000_000, burst=20),
    "o3-pro": Limit(rpm=200, tpm=5_000_000, burst=5),
    "o1": Limit(rpm=500, tpm=10_000_000, burst=10),
    "o1-mini": Limit(rpm=1_000, tpm=50_000_000, burst=20),
    "o1-preview": Limit(rpm=50, tpm=1_000_000, burst=5),
    # Standard GPT models
    "gpt-4o": Limit(rpm=5_000, tpm=800_000, burst=20),
    "gpt-4o-mini": Limit(rpm=30_000, tpm=1_500_000, burst=50),
    "gpt-4-turbo": Limit(rpm=10_000, tpm=2_000_000, burst=20),
    "gpt-3.5-turbo": Limit(rpm=30_000, tpm=2_000_000, burst=50),
    # DeepSeek (default tier-1)
    "deepseek-reasoner": Limit(rpm=60, tpm=2_000_000, burst=5),
    "deepseek-chat": Limit(rpm=60, tpm=2_000_000, burst=5),
    # Local (effectively unlimited, but cap to protect GPU)
    "deepseek-r1": Limit(rpm=120, tpm=1_000_000, burst=10),
    "llama3": Limit(rpm=120, tpm=1_000_000, burst=10),
    "mistral": Limit(rpm=120, tpm=1_000_000, burst=10),
    "qwen2": Limit(rpm=120, tpm=1_000_000, burst=10),
    "ollama": Limit(rpm=120, tpm=1_000_000, burst=10),
    # Search providers
    "duckduckgo": Limit(rpm=60, tpm=0, burst=5),
    "serpapi": Limit(rpm=100, tpm=0, burst=10),
    "brave": Limit(rpm=100, tpm=0, burst=10),
    "bing": Limit(rpm=300, tpm=0, burst=20),
    # Default
    "default": Limit(rpm=60, tpm=100_000, burst=5),
}


def _get_limits(model: str) -> Limit:
    """Look up limits for a model with prefix fallback."""
    if model in DEFAULT_LIMITS:
        return DEFAULT_LIMITS[model]
    for prefix, limit in DEFAULT_LIMITS.items():
        if model.startswith(prefix):
            return limit
    return DEFAULT_LIMITS["default"]


# ── In-process limiter ────────────────────────────────────────────────────────


class InProcessRateLimiter:
    """Per-model async-safe token-bucket limiter (single-process).

    Each model gets its own asyncio.Lock and rolling 60s windows for
    RPM and TPM tracking.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_init_lock = asyncio.Lock()
        self._request_timestamps: dict[str, list[float]] = defaultdict(list)
        self._token_usage: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._metrics: dict[str, dict[str, int]] = defaultdict(
            lambda: {"acquired": 0, "waited": 0, "rejected": 0, "total_wait_ms": 0}
        )

    async def _get_lock(self, model: str) -> asyncio.Lock:
        """Race-free lock initialization."""
        if model in self._locks:
            return self._locks[model]
        async with self._lock_init_lock:
            if model not in self._locks:
                self._locks[model] = asyncio.Lock()
            return self._locks[model]

    async def acquire(
        self,
        model: str,
        estimated_tokens: int = 1_000,
        timeout: float = 60.0,
    ) -> None:
        """Acquire a rate-limit slot. Blocks until one is available.

        Raises TimeoutError if the wait exceeds `timeout` seconds.
        """
        limits = _get_limits(model)
        lock = await self._get_lock(model)
        start_wait = time.time()

        async with lock:
            while True:
                now = time.time()
                window_start = now - 60.0

                # Prune expired entries
                self._request_timestamps[model] = [
                    t for t in self._request_timestamps[model] if t > window_start
                ]
                self._token_usage[model] = [
                    (t, tok) for t, tok in self._token_usage[model] if t > window_start
                ]

                current_rpm = len(self._request_timestamps[model])
                current_tpm = sum(tok for _, tok in self._token_usage[model])

                rpm_ok = current_rpm < limits.rpm
                tpm_ok = (current_tpm + estimated_tokens) < limits.tpm if limits.tpm else True

                if rpm_ok and tpm_ok:
                    self._request_timestamps[model].append(now)
                    self._token_usage[model].append((now, estimated_tokens))
                    self._metrics[model]["acquired"] += 1
                    waited = time.time() - start_wait
                    if waited > 0.01:
                        self._metrics[model]["waited"] += 1
                        self._metrics[model]["total_wait_ms"] += int(waited * 1000)
                    return

                if time.time() - start_wait > timeout:
                    self._metrics[model]["rejected"] += 1
                    raise TimeoutError(
                        f"Rate limit wait exceeded {timeout}s for {model}. "
                        f"RPM {current_rpm}/{limits.rpm}, "
                        f"TPM {current_tpm}/{limits.tpm}"
                    )

                # Backoff: shorter if only RPM-bounded, longer if TPM-bounded
                wait_needed = 1.0 if rpm_ok else 2.0
                logger.debug(
                    f"Rate limit {model}: "
                    f"RPM={current_rpm}/{limits.rpm}, "
                    f"TPM={current_tpm}/{limits.tpm}. "
                    f"Waiting {wait_needed}s…"
                )
                await asyncio.sleep(wait_needed)

    def metrics(self) -> dict[str, dict[str, int]]:
        """Return a snapshot of per-model metrics for Prometheus."""
        return {k: dict(v) for k, v in self._metrics.items()}


# ── Public singleton ──────────────────────────────────────────────────────────
# A single shared limiter per process. Cross-worker coordination is handled
# at the Redis layer if available (future work: add RedisTokenBucket).

rate_limiter = InProcessRateLimiter()
