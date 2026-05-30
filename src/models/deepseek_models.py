"""
DeepSeek model implementations.
Supports DeepSeek-R1 (reasoning) and DeepSeek-Chat via API.

DeepSeek-R1 exposes <think>...</think> blocks in the response
which contain the chain-of-thought reasoning trace.
"""

import re
import time
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI  # DeepSeek uses OpenAI-compatible API
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from loguru import logger

from src.models.base_model import (
    BaseModel, GenerationConfig, ModelResponse, ModelCapability
)
from src.utils.token_counter import TokenCounter, MODEL_CONTEXT_LIMITS
from src.utils.rate_limiter import rate_limiter
from config.settings import settings


DEEPSEEK_REASONING_MODELS = {"deepseek-reasoner", "deepseek-r1"}


class DeepSeekModel(BaseModel):
    """
    DeepSeek API model client.
    
    Handles the special <think> tag parsing for R1 reasoning models.
    DeepSeek's API is OpenAI-compatible so we reuse AsyncOpenAI client.
    """
    
    def __init__(self, model_name: str = "deepseek-reasoner"):
        super().__init__(model_name)
        
        self._client = AsyncOpenAI(
            api_key=settings.models.deepseek_api_key,
            base_url=settings.models.deepseek_base_url,
            timeout=600.0,  # R1 can take several minutes
            max_retries=0,
        )
        
        self._token_counter = TokenCounter(model_name)
        self._is_reasoning = any(
            model_name.startswith(m) for m in DEEPSEEK_REASONING_MODELS
        )
        
        self.capabilities = {ModelCapability.STREAMING}
        if self._is_reasoning:
            self.capabilities.add(ModelCapability.REASONING)
        
        logger.info(
            f"DeepSeek model initialized: {model_name} "
            f"(reasoning: {self._is_reasoning})"
        )
    
    def count_tokens(self, text: str) -> int:
        return self._token_counter.count_tokens(text)
    
    def get_context_limit(self) -> int:
        return MODEL_CONTEXT_LIMITS.get(self.model_name, 64_000)
    
    def _parse_reasoning_response(
        self,
        content: str
    ) -> tuple[str, Optional[str]]:
        """
        Parse DeepSeek-R1 response to extract <think> blocks.
        
        Returns (answer, reasoning_trace)
        """
        # Try structured reasoning_content field first (API v2)
        # If not available, parse <think> tags from content
        
        think_pattern = re.compile(
            r'<think>(.*?)</think>',
            re.DOTALL | re.IGNORECASE
        )
        
        reasoning_parts = think_pattern.findall(content)
        clean_content = think_pattern.sub('', content).strip()
        
        reasoning = '\n\n'.join(reasoning_parts) if reasoning_parts else None
        
        return clean_content, reasoning
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(4),
    )
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate with DeepSeek API."""
        config = config or GenerationConfig(
            max_tokens=8192,
            temperature=0.6,  # DeepSeek R1 recommends 0.5-0.7
        )
        
        # Validate context
        self.validate_messages(messages, config.max_tokens)
        
        # Rate limiting
        estimated_tokens = self._token_counter.count_messages_tokens(messages)
        await rate_limiter.acquire(self.model_name, estimated_tokens)
        
        params: dict = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        if config.stop_sequences:
            params["stop"] = config.stop_sequences
        
        start_time = time.time()
        
        try:
            response = await self._client.chat.completions.create(**params)
            latency_ms = (time.time() - start_time) * 1000
            
            choice = response.choices[0]
            raw_content = choice.message.content or ""
            
            # Extract reasoning from message or <think> tags
            reasoning_content = None
            if hasattr(choice.message, "reasoning_content"):
                reasoning_content = choice.message.reasoning_content
                content = raw_content
            else:
                content, reasoning_content = self._parse_reasoning_response(
                    raw_content
                )
            
            usage = response.usage
            logger.info(
                f"DeepSeek {self.model_name}: "
                f"{usage.prompt_tokens}pt + {usage.completion_tokens}ct "
                f"in {latency_ms:.0f}ms"
            )
            
            return ModelResponse(
                content=content,
                reasoning_content=reasoning_content,
                model=self.model_name,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                finish_reason=choice.finish_reason or "stop",
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream DeepSeek response, yielding only answer tokens."""
        config = config or GenerationConfig(max_tokens=8192, temperature=0.6)
        self.validate_messages(messages, config.max_tokens)
        
        await rate_limiter.acquire(self.model_name)
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
        }
        
        in_think_block = False
        buffer = ""
        
        try:
            async with await self._client.chat.completions.create(**params) as stream:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    
                    # Filter out <think> blocks from streamed output
                    buffer += delta
                    
                    while buffer:
                        if in_think_block:
                            end_idx = buffer.find("</think>")
                            if end_idx != -1:
                                buffer = buffer[end_idx + len("</think>"):]
                                in_think_block = False
                            else:
                                buffer = ""  # Still in think block
                        else:
                            start_idx = buffer.find("<think>")
                            if start_idx != -1:
                                yield buffer[:start_idx]
                                buffer = buffer[start_idx + len("<think>"):]
                                in_think_block = True
                            else:
                                yield buffer
                                buffer = ""
                                break
        except Exception as e:
            logger.error(f"DeepSeek streaming error: {e}")
            raise