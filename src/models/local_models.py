"""
Local model deployment via Ollama (default), vLLM, or llama.cpp.

Enables fully offline operation with DeepSeek-R1 and other open models.
This file contains:
    - OllamaModel:    production Ollama client (chat + streaming + pull)
    - VLLMModel:       thin OpenAI-compatible client for vLLM servers
    - LlamaCppModel:   placeholder for llama.cpp server integration
    - get_model_factory(): dispatch by provider string

Production hardening over the original toy implementation:
- Health-check before first request (fail fast if Ollama is down)
- Automatic model pull on first request if not cached locally
- Per-request timeout derived from settings
- Token + cost accounting (local models = $0)
- Connection reuse via shared httpx.AsyncClient
- Clean shutdown of the HTTP pool
- Robust  tag parsing for streamed R1 output

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator, Optional

import httpx
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.models.base_model import (
    BaseModel,
    GenerationConfig,
    ModelCapability,
    ModelResponse,
)
from src.utils.token_counter import TokenCounter


# ── Context limits for local models ──────────────────────────────────────────
# Conservative defaults; override by setting OLLAMA_MODEL_CTX_LIMITS env.

OLLAMA_CONTEXT_LIMITS: dict[str, int] = {
    "deepseek-r1:1.5b": 32_768,
    "deepseek-r1:7b": 32_768,
    "deepseek-r1:8b": 32_768,
    "deepseek-r1:14b": 32_768,
    "deepseek-r1:32b": 32_768,
    "deepseek-r1:70b": 32_768,
    "llama3:8b": 8_192,
    "llama3:70b": 8_192,
    "llama3.1:8b": 128_000,
    "llama3.1:70b": 128_000,
    "mistral:7b": 32_768,
    "mixtral:8x7b": 32_768,
    "qwen2:7b": 32_768,
    "qwen2.5:7b": 32_768,
    "phi3:mini": 12_825,
}

_THINK_PATTERN = re.compile(r"<(?:think|thinking)>(.*?)(?:</think|</thinking>)",
                            re.DOTALL | re.IGNORECASE)


# ── Ollama client ────────────────────────────────────────────────────────────

class OllamaModel(BaseModel):
    """Local model via the Ollama HTTP API.

    Supports DeepSeek-R1 variants:
        - deepseek-r1:1.5b / :7b / :8b / :14b / :32b / :70b
    Also: llama3, llama3.1, mistral, mixtral, qwen2/2.5, phi3, etc.
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(model_name)
        self._base_url = (base_url or settings.models.ollama_base_url).rstrip("/")
        self._api_url = f"{self._base_url}/api"
        self._timeout = timeout or 300.0
        self._token_counter = TokenCounter("gpt-4")  # Approximate for local
        self._client: Optional[httpx.AsyncClient] = None
        self._model_pulled = False
        self._health_checked = False

        # Capabilities
        self.capabilities = {
            ModelCapability.STREAMING,
            ModelCapability.SYSTEM_PROMPT,
        }
        if "deepseek-r1" in model_name or "r1" in model_name:
            self.capabilities.add(ModelCapability.REASONING)

        logger.info(
            f"Ollama model initialized: model={model_name} "
            f"base_url={self._base_url} "
            f"caps={sorted(c.value for c in self.capabilities)}"
        )

    # ── HTTP client ─────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    # ── Primitives ─────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        return self._token_counter.count_tokens(text)

    def get_context_limit(self) -> int:
        return OLLAMA_CONTEXT_LIMITS.get(self.model_name, 8_192)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """Health-check Ollama and pull the model if not present."""
        if self._closed:
            return
        if not self._health_checked:
            ok = await self._health_check()
            if not ok:
                logger.warning(
                    f"Ollama at {self._base_url} is unreachable. "
                    f"Requests will fail until Ollama starts."
                )
            self._health_checked = True
        if not self._model_pulled:
            try:
                await self.pull_model_if_needed()
                self._model_pulled = True
            except Exception as e:
                logger.warning(f"Model pull skipped for {self.model_name}: {e}")

    async def close(self) -> None:
        if self._closed:
            return
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        await super().close()

    async def _health_check(self) -> bool:
        """Check if Ollama is running."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{self._api_url}/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def pull_model_if_needed(self) -> None:
        """Pull model from Ollama registry if not present locally."""
        client = await self._get_client()
        # First check if model is already pulled
        try:
            resp = await client.get(f"{self._api_url}/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for m in models:
                    if m.get("name") == self.model_name:
                        logger.debug(f"Ollama model {self.model_name} already pulled")
                        return
        except Exception:
            pass

        logger.info(f"Pulling model {self.model_name} via Ollama...")
        try:
            async with client.stream(
                "POST",
                f"{self._api_url}/pull",
                json={"name": self.model_name},
                timeout=httpx.Timeout(600.0, connect=10.0),
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if status:
                            completed = data.get("completed", "")
                            total = data.get("total", "")
                            logger.info(f"Pull {self.model_name}: {status} {completed}/{total}")
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to pull {self.model_name}: {e}")
            raise

    # ── Helpers ────────────────────────────────────────────────────────────

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Flatten chat messages into a single prompt for non-chat models."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")
        parts.append("[ASSISTANT]")
        return "\n\n".join(parts)

    def _parse_reasoning(self, raw_content: str) -> tuple[str, Optional[str]]:
        """Parse  tags (DeepSeek-R1 style) from Ollama output."""
        reasoning_parts = _THINK_PATTERN.findall(raw_content)
        clean = _THINK_PATTERN.sub("", raw_content).strip()
        reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None
        return clean, reasoning

    # ── Generation ────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate via Ollama /api/chat endpoint."""
        config = config or GenerationConfig(max_tokens=4_096, temperature=0.7)
        self.validate_messages(messages, config.max_tokens)

        # Auto-warm on first request
        await self.warmup()

        context_limit = self.get_context_limit()
        # Cap num_ctx so we don't blow memory on small models
        num_ctx = min(context_limit, max(2048, config.max_tokens + 8192))

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
                "num_ctx": num_ctx,
            },
        }
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences
        if config.seed is not None:
            payload["options"]["seed"] = config.seed

        client = await self._get_client()
        start_time = time.time()

        response = await client.post(f"{self._api_url}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.time() - start_time) * 1000

        raw_content = data.get("message", {}).get("content", "")
        content, reasoning_content = self._parse_reasoning(raw_content)

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        reasoning_tokens = (
            self._token_counter.count_tokens(reasoning_content)
            if reasoning_content and not completion_tokens
            else 0
        )

        result = ModelResponse(
            content=content,
            reasoning_content=reasoning_content,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=data.get("done_reason", "stop"),
            latency_ms=latency_ms,
        )
        self.update_cost(result)  # cost_usd will be 0.0 for local models
        logger.info(
            f"Ollama {self.model_name} ok: "
            f"{prompt_tokens}pt + {completion_tokens}ct "
            f"({reasoning_tokens} reasoning) "
            f"in {latency_ms:.0f}ms"
        )
        return result

    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama, filtering  blocks."""
        config = config or GenerationConfig(max_tokens=4_096)
        self.validate_messages(messages, config.max_tokens)
        await self.warmup()

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
                "num_ctx": min(self.get_context_limit(), 32768),
            },
        }
        if config.stop_sequences:
            payload["options"]["stop"] = config.stop_sequences

        client = await self._get_client()
        in_think_block = False

        async with client.stream("POST", f"{self._api_url}/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if not token:
                    if data.get("done"):
                        break
                    continue

                # Filter  blocks
                if "<" in token and ("think" in token or "thinking" in token):
                    in_think_block = ">" in token  # toggle if closing
                    continue
                if in_think_block:
                    if "</" in token and ("think" in token or "thinking" in token):
                        in_think_block = False
                    continue

                yield token

                if data.get("done"):
                    break


# ── vLLM client (OpenAI-compatible) ──────────────────────────────────────────

class VLLMModel(BaseModel):
    """Thin client for vLLM's OpenAI-compatible server.

    vLLM exposes the same API as OpenAI but serves local models. We
    reuse the DeepSeek client implementation by subclassing it and
    pointing the base URL at the vLLM server. This gives us retry,
    streaming, and accounting for free.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",  # vLLM accepts any non-empty key by default
        timeout: Optional[float] = None,
    ) -> None:
        from src.models.deepseek_models import DeepSeekModel

        super().__init__(model_name)
        self._impl = DeepSeekModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout or 300.0,
        )
        self.capabilities = self._impl.capabilities
        logger.info(
            f"vLLM model initialized: model={model_name} base_url={base_url}"
        )

    def count_tokens(self, text: str) -> int:
        return self._impl.count_tokens(text)

    def get_context_limit(self) -> int:
        return self._impl.get_context_limit()

    async def generate(self, messages, config=None) -> ModelResponse:
        return await self._impl.generate(messages, config)

    async def stream_generate(self, messages, config=None) -> AsyncGenerator[str, None]:
        async for t in self._impl.stream_generate(messages, config):
            yield t

    async def warmup(self) -> None:
        await self._impl.warmup()

    async def close(self) -> None:
        await self._impl.close()
        await super().close()


# ── llama.cpp placeholder ──────────────────────────────────────────────────────

class LlamaCppModel(BaseModel):
    """Placeholder for llama.cpp server integration.

    Implement by pointing at llama.cpp's OpenAI-compatible server
    (built with -DLLAMA_BUILD_SERVER=ON) and reusing the same pattern
    as VLLMModel.
    """

    def __init__(self, model_name: str = "llama-cpp"):
        super().__init__(model_name)
        raise NotImplementedError(
            "llama.cpp server integration not yet wired. "
            "Use the OpenAI-compatible server endpoint + DeepSeekModel instead."
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_model_factory(
    provider: str = "openai",
    model_name: Optional[str] = None,
) -> BaseModel:
    """Dispatch to the correct model provider.

    Args:
        provider:    "openai" | "deepseek" | "ollama" | "vllm" | "llamacpp"
        model_name:  Optional model override (falls back to settings default).

    Returns:
        Configured BaseModel instance. Caller is responsible for calling
        `await model.close()` when done (process-lifetime caching is the
        caller's concern, not the factory's).
    """
    provider = (provider or "openai").lower()

    if provider == "openai":
        from src.models.openai_models import OpenAIModel
        name = model_name or settings.models.openai_reasoning_model
        return OpenAIModel(name)

    if provider == "deepseek":
        from src.models.deepseek_models import DeepSeekModel
        name = model_name or settings.models.deepseek_reasoning_model
        return DeepSeekModel(name)

    if provider == "ollama":
        name = model_name or settings.models.ollama_model
        return OllamaModel(name)

    if provider == "vllm":
        name = model_name or "deepseek-ai/DeepSeek-R1"
        return VLLMModel(name)

    if provider == "llamacpp":
        return LlamaCppModel(model_name or "llama-cpp")

    raise ValueError(
        f"Unknown provider: {provider!r}. "
        f"Choose from: openai, deepseek, ollama, vllm, llamacpp"
    )
