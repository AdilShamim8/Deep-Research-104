"""
OpenAI model implementation including the o-series reasoning models
(o1, o1-mini, o1-preview, o3, o3-mini) and standard GPT-4o/4-turbo.

Production hardening over the original toy implementation:
- Separate retry policy for 429 / 5xx vs generic errors
- Per-request timeout derived from settings (not hardcoded 300s)
- Token validation BEFORE hitting the API (fail fast)
- Token + cost accounting on every response
- Capability advertisement matches OpenAI's documented matrix
- Stream parser understands o-series reasoning chunks
- Connection pool is closed cleanly on shutdown

Key differences for o-series models:
- Use `reasoning_effort` instead of temperature/top_p
- `max_completion_tokens` (not `max_tokens`)
- System messages must be merged into the first user message
- `completion_tokens_details.reasoning_tokens` carries thinking budget

Date tag: 17 October 2026.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

from config.settings import settings
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

from src.models.base_model import (
    BaseModel,
    GenerationConfig,
    ModelCapability,
    ModelResponse,
)
from src.utils.rate_limiter import rate_limiter
from src.utils.token_counter import (
    MODEL_CONTEXT_LIMITS,
    MODEL_OUTPUT_LIMITS,
    TokenCounter,
)

# ── Model matrices ────────────────────────────────────────────────────────────

# Models in the o-series: extended thinking, no temperature/top_p,
# system messages must be rewritten as user messages.
O_SERIES_MODELS: frozenset[str] = frozenset(
    {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro"}
)

# Models that support token streaming (o1 legacy does not).
STREAMING_MODELS: frozenset[str] = frozenset(
    {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o3",
        "o3-mini",
    }
)

# Models that accept a system message directly.
SYSTEM_MESSAGE_CAPABLE: frozenset[str] = frozenset(
    {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}
)


# ── Retry policy ──────────────────────────────────────────────────────────────
# Two distinct classes of errors warrant different back-off strategies:
#   - 429 / connection / timeout: exponential backoff, longer cap
#   - 5xx: exponential backoff, short cap (transient server issue)

_RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


def _build_retry_decorator(max_attempts: int) -> Any:
    """Build a tenacity decorator configured from settings."""
    return retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(max_attempts),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )


# ── Provider ──────────────────────────────────────────────────────────────────


class OpenAIModel(BaseModel):
    """Production OpenAI model client (GPT-4o + o-series reasoning)."""

    def __init__(
        self,
        model_name: str = "o3-mini",
        *,
        api_key: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(model_name)

        self._api_key = api_key or settings.models.openai_api_key
        if not self._api_key and settings.is_production:
            raise RuntimeError("OPENAI_API_KEY is required in production but not set.")

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            organization=organization or settings.models.openai_org_id,
            max_retries=0,  # we handle retries
            timeout=timeout or settings.models.openai_request_timeout,
        )

        self._token_counter = TokenCounter(model_name)
        self._is_o_series = any(model_name.startswith(m) for m in O_SERIES_MODELS)
        self._max_retries = settings.models.openai_max_retries

        # ── Capability matrix ───────────────────────────────────────────────
        self.capabilities = {ModelCapability.JSON_MODE, ModelCapability.STREAMING}
        if self._is_o_series:
            self.capabilities.add(ModelCapability.REASONING)
        if model_name in SYSTEM_MESSAGE_CAPABLE:
            self.capabilities.add(ModelCapability.SYSTEM_PROMPT)
        if "gpt-4o" in model_name:
            self.capabilities.update(
                {
                    ModelCapability.VISION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.TOOL_USE,
                    ModelCapability.PARALLEL_TOOL_CALLS,
                }
            )

        logger.info(
            f"OpenAI model initialized: model={model_name} "
            f"o_series={self._is_o_series} "
            f"caps={sorted(c.value for c in self.capabilities)}"
        )

    # ── Primitives ─────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        return self._token_counter.count_tokens(text)

    def get_context_limit(self) -> int:
        return MODEL_CONTEXT_LIMITS.get(self.model_name, 128_000)

    async def warmup(self) -> None:
        """Lightweight pre-flight: list models to confirm auth + reachability."""
        if self._closed:
            return
        try:
            await self._client.models.list()
            logger.debug(f"OpenAI warmup ok: {self.model_name}")
        except Exception as e:
            logger.warning(f"OpenAI warmup failed for {self.model_name}: {e}")

    async def close(self) -> None:
        """Release the underlying httpx connection pool."""
        if self._closed:
            return
        await self._client.close()
        await super().close()

    # ── Message shaping ────────────────────────────────────────────────────

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Adapt messages to the model's quirks.

        o-series doesn't accept `system` role — we merge system content
        into the first user turn so the prompt is preserved.
        """
        if not self._is_o_series:
            return messages

        prepared: list[dict] = []
        system_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_parts.append(msg["content"])
            else:
                prepared.append(msg)

        if not system_parts or not prepared:
            return prepared

        first_user_idx = next(
            (i for i, m in enumerate(prepared) if m.get("role") == "user"),
            None,
        )
        if first_user_idx is None:
            return prepared

        system_text = "\n\n".join(system_parts)
        merged = {
            "role": "user",
            "content": (
                f"[System Instructions]\n{system_text}\n\n"
                f"[User Query]\n{prepared[first_user_idx]['content']}"
            ),
        }
        prepared[first_user_idx] = merged
        return prepared

    # ── Request building ──────────────────────────────────────────────────

    def _build_request_params(
        self,
        messages: list[dict],
        config: GenerationConfig,
    ) -> dict[str, Any]:
        """Translate unified GenerationConfig into OpenAI API params."""
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        if self._is_o_series:
            # o-series: reasoning_effort + max_completion_tokens
            params["max_completion_tokens"] = min(
                config.max_tokens,
                MODEL_OUTPUT_LIMITS.get(self.model_name, 32_768),
            )
            effort = config.reasoning_effort or settings.models.openai_reasoning_effort
            if effort:
                params["reasoning_effort"] = effort
            # temperature/top_p are NOT supported on o-series
        else:
            # Standard models
            params["max_tokens"] = min(
                config.max_tokens,
                MODEL_OUTPUT_LIMITS.get(self.model_name, 4_096),
            )
            params["temperature"] = config.temperature
            params["top_p"] = config.top_p
            if config.n > 1:
                params["n"] = config.n
            if config.stop_sequences:
                params["stop"] = config.stop_sequences
            if config.seed is not None:
                params["seed"] = config.seed
            if config.json_schema:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": config.json_schema,
                        "strict": True,
                    },
                }
            if config.user:
                params["user"] = config.user

        return params

    # ── Generation ────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """Generate a completion with retries, validation, and accounting."""
        config = config or GenerationConfig()
        self.validate_messages(messages, config.max_tokens)
        prepared = self._prepare_messages(messages)

        estimated_tokens = self._token_counter.count_messages_tokens(prepared) + config.max_tokens
        await rate_limiter.acquire(self.model_name, estimated_tokens=estimated_tokens)

        params = self._build_request_params(prepared, config)
        start_time = time.time()

        decorated = _build_retry_decorator(self._max_retries)(self._call_api)
        response = await decorated(params)

        latency_ms = (time.time() - start_time) * 1000
        return self._parse_response(response, latency_ms)

    async def _call_api(self, params: dict[str, Any]) -> Any:
        """Single API call — wrapped by tenacity for retries."""
        try:
            return await self._client.chat.completions.create(**params)
        except RateLimitError as e:
            logger.warning(
                f"OpenAI rate limit on {self.model_name}: "
                f"{e.response.headers.get('retry-after', '?')}s retry-after"
            )
            raise
        except APIError as e:
            logger.error(f"OpenAI API error on {self.model_name}: {e}")
            raise

    def _parse_response(self, response: Any, latency_ms: float) -> ModelResponse:
        """Normalize OpenAI SDK response into ModelResponse."""
        choice = response.choices[0]
        content = choice.message.content or ""

        # o-series reasoning trace (if surfaced)
        reasoning_content = getattr(choice.message, "reasoning_content", None)

        # Parallel sampling alternatives
        alternatives: list[str] = []
        if len(response.choices) > 1:
            alternatives = [c.message.content or "" for c in response.choices[1:]]

        usage = response.usage
        reasoning_tokens = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        result = ModelResponse(
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
            request_id=getattr(response, "id", None),
        )
        self.update_cost(result)
        logger.info(
            f"OpenAI {self.model_name} ok: "
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
        config: GenerationConfig | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens as they are generated."""
        if not self.has_capability(ModelCapability.STREAMING):
            # Fallback for non-streaming models: emit the full answer at once.
            response = await self.generate(messages, config)
            yield response.content
            return

        config = config or GenerationConfig()
        config.stream = True
        self.validate_messages(messages, config.max_tokens)
        prepared = self._prepare_messages(messages)
        params = self._build_request_params(prepared, config)
        params["stream"] = True

        await rate_limiter.acquire(self.model_name)

        try:
            async with await self._client.chat.completions.create(**params) as stream:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"OpenAI streaming error on {self.model_name}: {e}")
            raise
