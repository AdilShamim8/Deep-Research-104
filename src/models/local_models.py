"""
Local model deployment via Ollama, vLLM, or llama.cpp.
Enables fully offline operation with DeepSeek-R1 and other open models.
"""

import time
import json
import asyncio
from typing import AsyncGenerator, Optional
import httpx
from loguru import logger

from src.models.base_model import (
    BaseModel, GenerationConfig, ModelResponse, ModelCapability
)
from src.utils.token_counter import TokenCounter
from config.settings import settings


class OllamaModel(BaseModel):
    """
    Local model via Ollama API.
    
    Supports DeepSeek-R1 variants:
    - deepseek-r1:7b  (fast, low RAM)
    - deepseek-r1:14b (balanced)
    - deepseek-r1:32b (good quality)
    - deepseek-r1:70b (best, needs 64GB+ RAM)
    
    Also: llama3, mistral, phi3, qwen2, etc.
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        base_url: Optional[str] = None,
    ):
        super().__init__(model_name)
        self._base_url = base_url or settings.models.ollama_base_url
        self._api_url = f"{self._base_url}/api"
        self._token_counter = TokenCounter("gpt-4")  # Approx for local models
        
        self.capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.REASONING,
        }
        
        # Context limits for local models
        self._context_limits = {
            "deepseek-r1:7b": 32_768,
            "deepseek-r1:14b": 32_768,
            "deepseek-r1:32b": 32_768,
            "deepseek-r1:70b": 32_768,
            "llama3:8b": 8_192,
            "llama3:70b": 8_192,
            "mistral:7b": 32_768,
            "qwen2:7b": 32_768,
        }
    
    def count_tokens(self, text: str) -> int:
        return self._token_counter.count_tokens(text)
    
    def get_context_limit(self) -> int:
        return self._context_limits.get(self.model_name, 8_192)
    
    async def _health_check(self) -> bool:
        """Check if Ollama is running."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
            except Exception:
                return False
    
    async def pull_model_if_needed(self) -> None:
        """Pull model from Ollama registry if not present."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                logger.info(f"Pulling model {self.model_name} via Ollama...")
                async with client.stream(
                    "POST",
                    f"{self._api_url}/pull",
                    json={"name": self.model_name},
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "status" in data:
                                logger.info(
                                    f"Pull status: {data.get('status', '')} "
                                    f"{data.get('completed', '')}"
                                    f"/{data.get('total', '')}"
                                )
            except Exception as e:
                logger.error(f"Failed to pull {self.model_name}: {e}")
                raise
    
    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """
        Convert chat messages to a single prompt for models
        that don't support chat format natively.
        """
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            parts.append(f"[{role}]\n{content}")
        parts.append("[ASSISTANT]")
        return "\n\n".join(parts)
    
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate using Ollama /api/chat endpoint."""
        config = config or GenerationConfig(max_tokens=4096, temperature=0.7)
        
        # Check token limits
        token_count = self._token_counter.count_messages_tokens(messages)
        context_limit = self.get_context_limit()
        if token_count + config.max_tokens > context_limit:
            logger.warning(
                f"Reducing max_tokens: {token_count} input tokens + "
                f"{config.max_tokens} output > {context_limit} limit"
            )
            config.max_tokens = max(512, context_limit - token_count - 100)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
                "num_ctx": context_limit,
            }
        }
        
        start_time = time.time()
        
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0)
        ) as client:
            try:
                response = await client.post(
                    f"{self._api_url}/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                latency_ms = (time.time() - start_time) * 1000
                
                raw_content = data["message"]["content"]
                
                # Parse <think> blocks from DeepSeek-R1
                thinking_content = None
                import re
                think_match = re.search(
                    r'<think>(.*?)</think>',
                    raw_content,
                    re.DOTALL
                )
                if think_match:
                    thinking_content = think_match.group(1).strip()
                    raw_content = re.sub(
                        r'<think>.*?</think>', '',
                        raw_content,
                        flags=re.DOTALL
                    ).strip()
                
                # Extract token counts from Ollama response
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                
                logger.info(
                    f"Ollama {self.model_name}: "
                    f"{prompt_tokens}pt + {completion_tokens}ct "
                    f"in {latency_ms:.0f}ms"
                )
                
                return ModelResponse(
                    content=raw_content,
                    reasoning_content=thinking_content,
                    model=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    finish_reason=data.get("done_reason", "stop"),
                    latency_ms=latency_ms,
                )
                
            except httpx.HTTPError as e:
                logger.error(f"Ollama HTTP error: {e}")
                raise
    
    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream from Ollama API."""
        config = config or GenerationConfig(max_tokens=4096)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "num_ctx": self.get_context_limit(),
            }
        }
        
        in_think_block = False
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream(
                "POST",
                f"{self._api_url}/chat",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        
                        # Filter thinking tokens
                        if "<think>" in token:
                            in_think_block = True
                        if "</think>" in token:
                            in_think_block = False
                            token = token.split("</think>")[-1]
                        
                        if not in_think_block and token:
                            yield token
                            
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue


def get_model_factory(
    provider: str = "openai",
    model_name: Optional[str] = None,
) -> BaseModel:
    """
    Factory function to instantiate correct model provider.
    
    Args:
        provider: "openai", "deepseek", "ollama", "vllm"
        model_name: Specific model name override
    
    Returns:
        Configured BaseModel instance
    """
    if provider == "openai":
        from src.models.openai_models import OpenAIModel
        name = model_name or settings.models.openai_reasoning_model
        return OpenAIModel(name)
    
    elif provider == "deepseek":
        from src.models.deepseek_models import DeepSeekModel
        name = model_name or settings.models.deepseek_reasoning_model
        return DeepSeekModel(name)
    
    elif provider == "ollama":
        name = model_name or settings.models.ollama_model
        return OllamaModel(name)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: openai, deepseek, ollama"
        )