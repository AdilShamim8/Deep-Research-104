"""
Abstract base class for all LLM model providers.

Enforces a consistent interface across OpenAI, DeepSeek, and local
models (Ollama / vLLM / llama.cpp). Every concrete provider must
implement four primitives:

    - generate(...)        -> ModelResponse
    - stream_generate(...) -> AsyncGenerator[str, None]
    - count_tokens(text)   -> int
    - get_context_limit()  -> int

The base class adds shared production concerns:
    - Capability advertisement (reasoning / streaming / JSON mode / ...)
    - Message validation against context window (fail fast, not at API)
    - Lifecycle hooks for connection pools (warmup / teardown)
    - Per-provider cost estimation hook

Date tag: 17 October 2026.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Optional


# ── Capabilities ─────────────────────────────────────────────────────────────

class ModelCapability(Enum):
    """Capabilities a model may advertise."""
    REASONING = "reasoning"              # Extended thinking / CoT
    STREAMING = "streaming"              # Token streaming
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    JSON_MODE = "json_mode"
    TOOL_USE = "tool_use"
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    SYSTEM_PROMPT = "system_prompt"


# ── Generation config ─────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """
    Unified generation parameters. Provider adapters translate this
    into provider-specific request bodies.

    Attributes:
        max_tokens:           Maximum output tokens to generate.
        temperature:          Sampling temperature (0.0 - 2.0).
        top_p:                Nucleus sampling cutoff (0.0 - 1.0].
        stop_sequences:       Stop generation when any of these appear.
        stream:               If True, use stream_generate instead of generate.
        reasoning_effort:     OpenAI o-series: "low"|"medium"|"high".
        budget_tokens:        DeepSeek-R1 thinking budget.
        n:                    Number of parallel completions (parallel sampling).
        best_of:              Best-of-N server-side selection.
        seed:                 Deterministic sampling seed (when supported).
        json_schema:          Optional JSON schema for structured output.
        user:                 End-user identifier (for abuse monitoring).
        extra:                Provider-specific escape hatch.
    """
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False

    # Reasoning-specific
    reasoning_effort: Optional[str] = None       # "low" | "medium" | "high"
    budget_tokens: Optional[int] = None         # DeepSeek-R1 thinking budget

    # Sampling
    n: int = 1
    best_of: Optional[int] = None
    seed: Optional[int] = None

    # Structured output
    json_schema: Optional[dict[str, Any]] = None

    # Telemetry / safety
    user: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Defensive bounds-checking at construction time."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0.0, 1.0]")
        if self.n < 1:
            raise ValueError("n must be >= 1")


# ── Response ──────────────────────────────────────────────────────────────────

@dataclass
class ModelResponse:
    """Unified response structure from any model provider."""
    content: str                                  # Final answer
    reasoning_content: Optional[str] = None        # Chain-of-thought / thinking trace
    model: str = ""

    # Usage stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    # For parallel sampling (when n > 1)
    alternatives: list[str] = field(default_factory=list)

    # Metadata
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    cached: bool = False
    request_id: Optional[str] = None

    # Cost — populated by the provider via update_cost()
    cost_usd: float = 0.0

    @property
    def total_cost_estimate(self) -> float:
        """Backward-compat alias for cost_usd."""
        return self.cost_usd

    def to_log_dict(self) -> dict[str, Any]:
        """Project a log-safe summary (no content) for observability."""
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 2),
            "finish_reason": self.finish_reason,
            "cached": self.cached,
            "cost_usd": round(self.cost_usd, 6),
        }


# ── Pricing registry (per-provider overrides) ─────────────────────────────────
# Cheap lookup table for cost estimation. Override per provider by
# subclassing and setting self._pricing.

PRICING_TABLE: dict[str, dict[str, float]] = {
    # Per 1K tokens (USD). Update as pricing evolves.
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "o3": {"input": 0.015, "output": 0.060},
    "o1": {"input": 0.015, "output": 0.060},
    "deepseek-reasoner": {"input": 0.00055, "output": 0.00219},
    "deepseek-chat": {"input": 0.00014, "output": 0.00028},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a generation. Returns 0.0 for unknown models."""
    pricing = PRICING_TABLE.get(model)
    if not pricing:
        # Try a prefix match (e.g., "o3-mini-2025-01-31" -> "o3-mini")
        for known in PRICING_TABLE:
            if model.startswith(known):
                pricing = PRICING_TABLE[known]
                break
    if not pricing:
        return 0.0
    return (
        (prompt_tokens / 1000.0) * pricing["input"]
        + (completion_tokens / 1000.0) * pricing["output"]
    )


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseModel(ABC):
    """
    Abstract interface for all LLM providers.

    Concrete implementations must handle:
      - Token limit validation before calling the API (fail fast)
      - Rate limiting (via src.utils.rate_limiter)
      - Retry logic with exponential backoff (via tenacity)
      - Response normalization to ModelResponse
      - Cost estimation via update_cost()
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.capabilities: set[ModelCapability] = set()
        self._closed: bool = False

    # ── Required primitives ─────────────────────────────────────────────────

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate a response from messages."""

    @abstractmethod
    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generation token by token."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for this model."""

    @abstractmethod
    def get_context_limit(self) -> int:
        """Return max context window in tokens."""

    # ── Lifecycle hooks ────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """Optional warmup hook (e.g., pre-download tokenizer, ping host).

        Default implementation is a no-op. Override to do provider-specific
        pre-flight checks. Safe to call multiple times.
        """
        return None

    async def close(self) -> None:
        """Release connection pools / file handles. Idempotent."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ── Capability checks ───────────────────────────────────────────────────

    def has_capability(self, cap: ModelCapability) -> bool:
        return cap in self.capabilities

    def require_capability(self, cap: ModelCapability) -> None:
        """Raise if the model doesn't advertise a capability."""
        if not self.has_capability(cap):
            raise NotImplementedError(
                f"Model {self.model_name!r} does not support {cap.value!r}. "
                f"Advertised: {sorted(c.value for c in self.capabilities)}"
            )

    # ── Validation ─────────────────────────────────────────────────────────

    def validate_messages(
        self,
        messages: list[dict],
        reserved_output_tokens: int = 2048,
    ) -> None:
        """Validate messages fit in context. Raises ValueError if not."""
        from src.utils.token_counter import TokenCounter

        counter = TokenCounter(self.model_name)
        fits, count, available = counter.check_fits_in_context(
            messages, reserved_output_tokens
        )
        if not fits:
            raise ValueError(
                f"Messages ({count} tokens) exceed context limit "
                f"({self.get_context_limit()}) for {self.model_name}. "
                f"Reduce input by "
                f"{count - self.get_context_limit() + reserved_output_tokens} "
                f"tokens."
            )

    # ── Cost tracking ───────────────────────────────────────────────────────

    def update_cost(self, response: ModelResponse) -> None:
        """Populate response.cost_usd based on token usage."""
        response.cost_usd = estimate_cost(
            self.model_name,
            response.prompt_tokens,
            response.completion_tokens,
        )

    # ── Convenience ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        caps = sorted(c.value for c in self.capabilities)
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"capabilities={caps})"
        )
