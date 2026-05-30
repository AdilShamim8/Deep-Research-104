"""
Token bucket rate limiter for API calls.
Prevents hitting provider rate limits in production.
"""

import asyncio
import time
from collections import defaultdict
from typing import Optional
from loguru import logger


class TokenBucketRateLimiter:
    """
    Async-safe token bucket rate limiter.
    
    Supports per-model rate limiting since different models
    have different rate limits (RPM, TPM).
    """
    
    # OpenAI rate limits (requests per minute, tokens per minute)
    DEFAULT_LIMITS = {
        "o3": {"rpm": 500, "tpm": 10_000_000},
        "o3-mini": {"rpm": 1_000, "tpm": 50_000_000},
        "o1": {"rpm": 500, "tpm": 10_000_000},
        "o1-mini": {"rpm": 1_000, "tpm": 50_000_000},
        "gpt-4o": {"rpm": 5_000, "tpm": 800_000},
        "deepseek-reasoner": {"rpm": 60, "tpm": 2_000_000},
        "deepseek-chat": {"rpm": 60, "tpm": 2_000_000},
        "default": {"rpm": 60, "tpm": 100_000},
    }
    
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._request_timestamps: dict[str, list[float]] = defaultdict(list)
        self._token_timestamps: dict[str, list[tuple[float, int]]] = defaultdict(list)
    
    def _get_limits(self, model: str) -> dict:
        """Get rate limits for a model."""
        for key in self.DEFAULT_LIMITS:
            if model.startswith(key):
                return self.DEFAULT_LIMITS[key]
        return self.DEFAULT_LIMITS["default"]
    
    async def acquire(
        self,
        model: str,
        estimated_tokens: int = 1000,
        timeout: float = 60.0
    ) -> None:
        """
        Acquire rate limit slot. Blocks until slot is available.
        Raises TimeoutError if wait exceeds timeout.
        """
        limits = self._get_limits(model)
        rpm_limit = limits["rpm"]
        tpm_limit = limits["tpm"]
        
        start_wait = time.time()
        
        async with self._locks[model]:
            while True:
                now = time.time()
                window_start = now - 60.0
                
                # Clean old timestamps
                self._request_timestamps[model] = [
                    t for t in self._request_timestamps[model]
                    if t > window_start
                ]
                self._token_timestamps[model] = [
                    (t, tok) for t, tok in self._token_timestamps[model]
                    if t > window_start
                ]
                
                # Check RPM
                current_rpm = len(self._request_timestamps[model])
                # Check TPM
                current_tpm = sum(
                    tok for _, tok in self._token_timestamps[model]
                )
                
                rpm_ok = current_rpm < rpm_limit
                tpm_ok = (current_tpm + estimated_tokens) < tpm_limit
                
                if rpm_ok and tpm_ok:
                    # Record this request
                    self._request_timestamps[model].append(now)
                    self._token_timestamps[model].append(
                        (now, estimated_tokens)
                    )
                    return
                
                # Check timeout
                if time.time() - start_wait > timeout:
                    raise TimeoutError(
                        f"Rate limit wait exceeded {timeout}s for {model}. "
                        f"RPM: {current_rpm}/{rpm_limit}, "
                        f"TPM: {current_tpm}/{tpm_limit}"
                    )
                
                wait_needed = 1.0 if rpm_ok else 2.0
                logger.debug(
                    f"Rate limiting {model}: "
                    f"RPM={current_rpm}/{rpm_limit}, "
                    f"TPM={current_tpm}/{tpm_limit}. "
                    f"Waiting {wait_needed}s..."
                )
                await asyncio.sleep(wait_needed)


# Global rate limiter singleton
rate_limiter = TokenBucketRateLimiter()