"""
DeepSeek model implementation (R1 reasoning + DeepSeek-Chat).

DeepSeek-R1 exposes a `reasoning_content` field on the message (API v2)
or, for backward-compat clients, embeds the chain-of-thought trace` tags inside the regular `content` field.
We handle both shapes and surface the trace as `ModelResponse.reasoning_content`.

Production hardening over the toy original:
- Per-request timeout derived from settings (R1 can take minutes)
- Separate retry policy for 429 / 5xx vs generic errors
- Token + cost accounting on every response
- Stream parser correctly filters `<think>` blocks while preserving
  answer tokens that arrive mid-stream (the original buffer logic
  dropped the final partial chunk)
- Connection pool is closed cleanly on shutdown
- Capability advertisement matches DeepSeek's documented matrix

Date tag: 17 October 2026.
"""

from __future__ import annotations

import re
import time
from typing import Any, AsyncGenerator, Optional

from loguru import logger
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
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
from src.utils.rate_limiter import rate_limiter
from src.utils.token_counter import MODEL_CONTEXT_LIMITS, TokenCounter


# ── Model matrix ──────────────────────────────────────────────────────────────

DEEPSEEK_REASONING_MODELS: frozenset[str] = frozenset(
    {"deepseek-reasoner", "deepseek-r1"}
)

# DeepSeek-R1 recommends temperature 0.5-0.7 (0.6 is the documented default).
DEEPSEEK_R1_DEFAULT_TEMP = 0.6


# ── Reasoning parser ──────────────────────────────────────────────────────────

_THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)


def parse_reasoning_response(content: str) -> tuple[str, Optional[str]]:
    """Parse `<think>...</think>` blocks out of a DeepSeek-R1 response.

    Returns (clean_answer, reasoning_trace). reasoning_trace is None
    when the response didn't include any `<think>` blocks.
    """
    reasoning_parts = _THINK_PATTERN.findall(content)
    clean_content = _THINK_PATTERN.sub("", content).strip()
    reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None
    return clean_content, reasoning


# ── Retry policy ──────────────────────────────────────────────────────────────

_RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


def _build_retry_decorator(max_attempts: int) -> Any:
    """Build a tenacity decorator. R1 has long latency so cap backoff at 120s."""
    return retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )


# ── Provider ──────────────────────────────────────────────────────────────────

class DeepSeekModel(BaseModel):
    """DeepSeek API client (OpenAI-compatible). Supports R1 reasoning + chat."""

    def __init__(
        self,
        model_name: str = "deepseek-reasoner",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(model_name)

        self._api_key = api_key or settings.models.deepseek_api_key
        if not self._api_key and settings.is_production:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is required in production but not set."
            )

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url or settings.models.deepseek_base_url,
            max_retries=0,
            timeout=timeout or settings.models.deepseek_request_timeout,
        )

        self._token_counter = TokenCounter(model_name)
        self._is_reasoning = any(
            model_name.startswith(m) for m in DEEPSEEK_REASONING_MODELS
        )
        self._max_retries = settings.models.deepseek_max_retries

        # DeepSeek advertises: streaming, reasoning (R1 only), system prompt
        self.capabilities = {ModelCapability.STREAMING, ModelCapability.SYSTEM_PROMPT}
        if self._is_reasoning:
            self.capabilities.add(ModelCapability.REASONING)

        logger.info(
            f"DeepSeek model initialized: model={model_name} "
            f"reasoning={self._is_reasoning} "
            f"caps={sorted(c.value for c in self.capabilities)}"
        )

    # ── Primitives ─────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        return self._token_counter.count_tokens(text)

    def get_context_limit(self) -> int:
        return MODEL_CONTEXT_LIMITS.get(self.model_name, 64_000)

    async def warmup(self) -> None:
        """Pre-flight check: hit a tiny endpoint to validate auth/reachability."""
        if self._closed:
            return
        try:
            await self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            logger.debug(f"DeepSeek warmup ok: {self.model_name}")
        except Exception as e:
            logger.warning(f"DeepSeek warmup failed for {self.model_name}: {e}")

    async def close(self) -> None:
        if self._closed:
            return
        await self._client.close()
        await super().close()

    # ── Request building ──────────────────────────────────────────────────

    def _build_request_params(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature if not self._is_reasoning
                          else DEEPSEEK_R1_DEFAULT_TEMP,
            "top_p": config.top_p,
        }
        if config.stop_sequences:
            params["stop"] = config.stop_sequences
        if config.seed is not None:
            params["seed"] = config.seed
        if config.json_schema:
            # DeepSeek supports response_format json_object (no strict schema)
            params["response_format"] = {"type": "json_object"}
        if config.user:
            params["user"] = config.user
        return params

    # ── Generation ────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> ModelResponse:
        """Generate a DeepSeek completion with retries + accounting."""
        if config is None:
            config = GenerationConfig(
                max_tokens=8_192,
                temperature=DEEPSEEK_R1_DEFAULT_TEMP if self._is_reasoning else 0.7,
            )

        self.validate_messages(messages, config.max_tokens)

        estimated_tokens = (
            self._token_counter.count_messages_tokens(messages) + config.max_tokens
        )
        await rate_limiter.acquire(self.model_name, estimated_tokens=estimated_tokens)

        params = self._build_request_params(messages, config)
        start_time = time.time()

        decorated = _build_retry_decorator(self._max_retries)(self._call_api)
        response = await decorated(params)

        latency_ms = (time.time() - start_time) * 1000
        return self._parse_response(response, latency_ms)

    async def _call_api(self, params: dict[str, Any]) -> Any:
        """Single API call, wrapped by tenacity."""
        try:
            return await self._client.chat.completions.create(**params)
        except RateLimitError as e:
            retry_after = e.response.headers.get("retry-after", "?")
            logger.warning(
                f"DeepSeek rate limit on {self.model_name}: "
                f"retry-after={retry_after}s"
            )
            raise
        except APIError as e:
            logger.error(f"DeepSeek API error on {self.model_name}: {e}")
            raise

    def _parse_response(self, response: Any, latency_ms: float) -> ModelResponse:
        """Normalize DeepSeek SDK response into ModelResponse.

        DeepSeek-R1 surfaces reasoning either via a dedicated
        `reasoning_content` field (API v2) or via `<think>...</think>` tags
        embedded in `content` (v1 / older clients). Handle both.
        """
        choice = response.choices[0]
        raw_content = choice.message.content or ""

        reasoning_content = getattr(choice.message, "reasoning_content", None)
        if reasoning_content:
            content = raw_content
        else:
            content, reasoning_content = parse_reasoning_response(raw_content)

        usage = response.usage
        reasoning_tokens = 0
        # DeepSeek reports reasoning_tokens via completion_tokens_details
        # on R1; older SDKs may omit this.
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        # Fallback: if we extracted reasoning text, approximate its token count
        if not reasoning_tokens and reasoning_content:
            reasoning_tokens = self._token_counter.count_tokens(reasoning_content)

        result = ModelResponse(
            content=content,
            reasoning_content=reasoning_content,
            model=self.model_name,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=usage.total_tokens,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency_ms,
            request_id=getattr(response, "id", None),
        )
        self.update_cost(result)
        logger.info(
            f"DeepSeek {self.model_name} ok: "
            f"{usage.prompt_tokens}pt + {usage.completion_tokens}ct "
            f"({reasoning_tokens} reasoning) "
            f"in {latency_ms:.0f}ms "
            f"cost=${result.cost_usd:.6f}"
        )
        return result

    # ── Streaming ──────────────────────────────────────────────────────────

    async def stream_generate(
        self,
        messages: list[dict],
        config: Optional[GenerationConfig] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream DeepSeek answer tokens, filtering `<think>` blocks.

        R1 streams reasoning inside `<think>...</think>` blocks before
        yielding answer tokens. We suppress reasoning chunks and only
        forward answer text to the caller.
        """
        if config is None:
            config = GenerationConfig(
                max_tokens=8_192,
                temperature=DEEPSEEK_R1_DEFAULT_TEMP if self._is_reasoning else 0.7,
            )
        self.validate_messages(messages, config.max_tokens)

        await rate_limiter.acquire(self.model_name)

        params = self._build_request_params(messages, config)
        params["stream"] = True

        in_think_block = False
        buffer = ""

        try:
            async with await self._client.chat.completions.create(**params) as stream:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not delta or not delta.content:
                        continue

                    buffer += delta.content

                    # Process buffer: emit answer tokens, hold <think> blocks
                    while buffer:
                        if in_think_block:
                            end_idx = buffer.find("</think>")
                            if end_idx != -1:
                                buffer = buffer[end_idx + len("</think>"):]
                                in_think_block = False
                                continue
                            buffer = ""
                            break
                        else:
                            start_idx = buffer.find("<think>")
                            if start_idx != -1:
                                if start_idx > 0:
                                    yield buffer[:start_idx]
                                buffer = buffer[start_idx + len("<think>"):]
                                in_think_block = True
                                continue
                            # No <think> tag yet; hold last 7 chars in case
                            # the tag spans chunks.
                            if len(buffer) > 7:
                                yield buffer[:-7]
                                buffer = buffer[-7:]
                            break
            # Flush remaining answer buffer (no more chunks coming)
            if buffer and not in_think_block:
                yield buffer
        except Exception as e:
            logger.error(f"DeepSeek streaming error on {self.model_name}: {e}")
            raise
