"""
OpenAI model implementations including o1, o3 reasoning models.

Key differences for o-series models:
- Use 'reasoning_effort' instead of temperature
- No system message (use user message instead)
- Higher max_completion_tokens limits
- Includes reasoning_tokens in usage
"""

import time
import asyncio
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI, RateLimitError, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger

from src.models.base_model import (
    BaseModel, GenerationConfig, ModelResponse, ModelCapability
)
from src.utils.token_counter import (
    TokenCounter, MODEL_CONTEXT_LIMITS, MODEL_OUTPUT_LIMITS
)
from src.utils.rate_limiter import rate_limiter
from config.settings import settings


# O-series models require special handling
O_SERIES_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini"}

# Models that support streaming
STREAMING_MODELS = {"gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "o3", "o3-mini"}


class OpenAIModel(BaseModel):
    """
    Production OpenAI model client.
    Handles both standard GPT models and reasoning o-series.
    """
    
    def __init__(self, model_name: str = "o3-mini"):
        super().__init__(model_name)
        
        self._client = AsyncOpenAI(
            api_key=settings.models.openai_api_key,
            organization=settings.models.openai_org_id,
            max_retries=0,  # We handle retries manually with tenacity
            timeout=300.0,  # 5 min timeout for reasoning models
        )
        
        self._token_counter = TokenCounter(model_name)
        self._is_o_series = any(
            model_name.startswith(m) for m in O_SERIES_MODELS
        )
        
        # Set capabilities
        self.capabilities = {ModelCapability.JSON_MODE}
        if self._is_o_series:
            self.capabilities.add(ModelCapability.REASONING)
        if model_name in STREAMING_MODELS:
            self.capabilities.add(ModelCapability.STREAMING)
        if "gpt-4o" in model_name:
            self.capabilities.add(ModelCapability.VISION)
        
        logger.info(
            f"OpenAI model initialized: {model_name} "
            f"(o-series: {self._is_o_series})"
        )
    
    def count_tokens(self, text: str) -> int:
        return self._token_counter.count_tokens(text)
    
    def get_context_limit(self) -> int:
        return MODEL_CONTEXT_LIMITS.get(self.model_name, 128_000)
    
    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """
        Prepare messages for o-series models.
        O-series don't support system messages - convert to user message.
        """
        if not self._is_o_series:
            return messages
        
        prepared = []
        system_content = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content.append(msg["content"])
            else:
                prepared.append(msg)
        
        # Prepend system content to first user message
        if system_content and prepared:
            first_user_idx = next(
                (i for i, m in enumerate(prepared) if m["role"] == "user"),
                None
            )
            if first_user_idx is not None:
                system_text = "\n\n".join(system_content)
                prepared[first_user_idx] = {
                    "role": "user",
                    "content": f"[System Instructions]\n{system_text}\n\n"
                               f"[User Query]\n{prepared[first_user_idx]['content']}"
                }
        
        return prepared
    
    def _build_request_params(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> dict:
        """Build API request parameters based on model type."""
        params = {
            "model": self.model_name,
            "messages": messages,
        }
        
        if self._is_o_series:
            # O-series specific parameters
            params["max_completion_tokens"] = min(
                config.max_tokens,
                MODEL_OUTPUT_LIMITS.get(self.model_name, 32_768)
            )
            if config.reasoning_effort:
                params["reasoning_effort"] = config.reasoning_effort
            # O-series don't support temperature/top_p
        else:
            # Standard model parameters
            params["max_tokens"] = min(
                config.max_tokens,
                MODEL_OUTPUT_LIMITS.get(self.model_name, 4_096)
            )
            params["temperature"] = config.temperature
            params["top_p"] = config.top_p
            
            if config.n > 1:
                params["n"] = config.n
            if config.stop_sequences:
                params["stop"] = config.stop_sequences
        
        return params
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """
        Generate completion with full retry logic and token validation.
        """
        config = config or GenerationConfig()
        
        # Validate token limits
        self.validate_messages(messages, config.max_tokens)
        
        # Prepare messages for model type
        prepared_messages = self._prepare_messages(messages)
        
        # Estimate tokens for rate limiting
        estimated_tokens = self._token_counter.count_messages_tokens(
            prepared_messages
        ) + config.max_tokens
        
        # Wait for rate limit slot
        await rate_limiter.acquire(
            self.model_name,
            estimated_tokens=estimated_tokens
        )
        
        # Build request
        params = self._build_request_params(prepared_messages, config)
        
        start_time = time.time()
        
        try:
            response = await self._client.chat.completions.create(**params)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract primary content
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Extract reasoning content (o-series)
            reasoning_content = None
            if hasattr(choice.message, "reasoning_content"):
                reasoning_content = choice.message.reasoning_content
            
            # Extract alternatives for parallel sampling
            alternatives = []
            if len(response.choices) > 1:
                alternatives = [
                    c.message.content or ""
                    for c in response.choices[1:]
                ]
            
            # Usage stats
            usage = response.usage
            reasoning_tokens = 0
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                reasoning_tokens = getattr(
                    usage.completion_tokens_details,
                    "reasoning_tokens",
                    0
                ) or 0
            
            logger.info(
                f"OpenAI {self.model_name} response: "
                f"{usage.prompt_tokens}pt + {usage.completion_tokens}ct "
                f"({reasoning_tokens} reasoning) in {latency_ms:.0f}ms"
            )
            
            return ModelResponse(
                content=content,
                reasoning_content=reasoning_content,
                model=self.model_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=usage.total_tokens,
                alternatives=alternatives,
                finish_reason=choice.finish_reason or "stop",
                latency_ms=latency_ms,
            )
            
        except RateLimitError as e:
            logger.error(f"Rate limit hit for {self.model_name}: {e}")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error for {self.model_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling {self.model_name}: {e}")
            raise
    
    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens as they're generated."""
        if not self.has_capability(ModelCapability.STREAMING):
            # Fallback: generate full response and yield at once
            response = await self.generate(messages, config)
            yield response.content
            return
        
        config = config or GenerationConfig(stream=True)
        config.stream = True
        
        self.validate_messages(messages, config.max_tokens)
        prepared_messages = self._prepare_messages(messages)
        params = self._build_request_params(prepared_messages, config)
        params["stream"] = True
        
        await rate_limiter.acquire(self.model_name)
        
        try:
            async with await self._client.chat.completions.create(
                **params
            ) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise