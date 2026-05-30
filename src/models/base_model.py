"""
Abstract base class for all LLM model providers.
Enforces consistent interface across OpenAI, DeepSeek, and local models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional
from enum import Enum


class ModelCapability(Enum):
    """Capabilities a model may support."""
    REASONING = "reasoning"           # Extended thinking / CoT
    STREAMING = "streaming"           # Token streaming
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    JSON_MODE = "json_mode"


@dataclass
class GenerationConfig:
    """
    Unified generation parameters.
    Maps to model-specific parameters internally.
    """
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False
    
    # Reasoning-specific
    reasoning_effort: Optional[str] = None   # "low", "medium", "high" (OpenAI o-series)
    budget_tokens: Optional[int] = None       # DeepSeek-R1 thinking budget
    
    # Sampling
    n: int = 1                               # Number of completions (parallel sampling)
    best_of: Optional[int] = None


@dataclass  
class ModelResponse:
    """Unified response structure from any model."""
    content: str                              # Final answer
    reasoning_content: Optional[str] = None  # Chain-of-thought / thinking trace
    model: str = ""
    
    # Usage stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    
    # For parallel sampling
    alternatives: list[str] = field(default_factory=list)
    
    # Metadata
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    cached: bool = False
    
    @property
    def total_cost_estimate(self) -> float:
        """Rough cost estimate - override per provider."""
        return 0.0


class BaseModel(ABC):
    """
    Abstract interface for all LLM providers.
    
    All concrete implementations must handle:
    - Token limit validation before calling API
    - Rate limiting via rate_limiter
    - Retry logic with exponential backoff
    - Response normalization to ModelResponse
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.capabilities: set[ModelCapability] = set()
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate a response from messages."""
        ...
    
    @abstractmethod
    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generation token by token."""
        ...
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens for this model."""
        ...
    
    @abstractmethod
    def get_context_limit(self) -> int:
        """Return max context window in tokens."""
        ...
    
    def has_capability(self, cap: ModelCapability) -> bool:
        return cap in self.capabilities
    
    def validate_messages(
        self,
        messages: list[dict],
        reserved_output_tokens: int = 2048,
    ) -> None:
        """
        Validate messages fit in context.
        Raises ValueError if they don't.
        """
        from src.utils.token_counter import TokenCounter
        counter = TokenCounter(self.model_name)
        fits, count, available = counter.check_fits_in_context(
            messages, reserved_output_tokens
        )
        if not fits:
            raise ValueError(
                f"Messages ({count} tokens) exceed context limit "
                f"({self.get_context_limit()}) for {self.model_name}. "
                f"Reduce input by {count - self.get_context_limit() + reserved_output_tokens} tokens."
            )