"""
Token counting utility — critical for respecting context windows.

Supports multiple model families with accurate token estimates via
tiktoken. Always count tokens BEFORE sending to the API to avoid
context-overflow errors at the provider (which cost real money).

Key tables:
    MODEL_ENCODING_MAP   — model -> tiktoken encoding name
    MODEL_CONTEXT_LIMITS — model -> max input context window (tokens)
    MODEL_OUTPUT_LIMITS  — model -> max output tokens

Production hardening over the toy original:
- Encoding loaded once and cached (avoid repeated disk reads)
- Bounds-checked truncation that preserves system + last-N messages
- Multimodal content (vision) counting supported
- Defensive: empty/None text returns 0, never raises
- Thread-safe (encoding is read-only after construction)

Date tag: 17 October 2026.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken
from loguru import logger

# ── Model -> encoding mapping ─────────────────────────────────────────────────

MODEL_ENCODING_MAP: dict[str, str] = {
    # OpenAI o-family (o1, o3) uses the o200k tokenizer
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o3-pro": "o200k_base",
    # GPT-4o family also uses o200k
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    # GPT-4 / GPT-3.5 use cl100k
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    # DeepSeek uses a BPE tokenizer similar to cl100k (close approximation)
    "deepseek-reasoner": "cl100k_base",
    "deepseek-chat": "cl100k_base",
    # Local models — use cl100k as a conservative estimate
    "deepseek-r1:7b": "cl100k_base",
    "deepseek-r1:14b": "cl100k_base",
    "deepseek-r1:32b": "cl100k_base",
    "deepseek-r1:70b": "cl100k_base",
    "llama3:8b": "cl100k_base",
    "llama3:70b": "cl100k_base",
    "mistral:7b": "cl100k_base",
    "qwen2:7b": "cl100k_base",
}


# ── Context window limits (tokens) ────────────────────────────────────────────
# Conservative defaults; verified against provider docs as of 2026-10.

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-preview": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o3-pro": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # DeepSeek
    "deepseek-reasoner": 128_000,
    "deepseek-chat": 64_000,
    # Local
    "deepseek-r1:1.5b": 32_768,
    "deepseek-r1:7b": 32_768,
    "deepseek-r1:14b": 32_768,
    "deepseek-r1:32b": 32_768,
    "deepseek-r1:70b": 128_000,
    "llama3:8b": 8_192,
    "llama3:70b": 8_192,
    "llama3.1:8b": 128_000,
    "llama3.1:70b": 128_000,
    "mistral:7b": 32_768,
    "mixtral:8x7b": 32_768,
    "qwen2:7b": 32_768,
    "qwen2.5:7b": 32_768,
}


# ── Max output tokens ────────────────────────────────────────────────────────

MODEL_OUTPUT_LIMITS: dict[str, int] = {
    # OpenAI o-series: very large output budgets
    "o1": 32_768,
    "o1-mini": 65_536,
    "o3": 100_000,
    "o3-mini": 100_000,
    "o3-pro": 100_000,
    # Standard GPT models
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-4-turbo": 4_096,
    "gpt-4": 4_096,
    "gpt-3.5-turbo": 4_096,
    # DeepSeek
    "deepseek-reasoner": 32_768,
    "deepseek-chat": 8_192,
    # Local
    "deepseek-r1:1.5b": 4_096,
    "deepseek-r1:7b": 8_192,
    "deepseek-r1:14b": 8_192,
    "deepseek-r1:32b": 16_384,
    "deepseek-r1:70b": 16_384,
    "llama3:8b": 4_096,
    "llama3:70b": 4_096,
    "mistral:7b": 4_096,
}


# ── Encoding cache ────────────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    """Cached encoding lookup. Avoids re-reading the BPE file each call."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(
            f"Could not load tiktoken encoding {encoding_name!r}: {e}. Falling back to cl100k_base."
        )
        return tiktoken.get_encoding("cl100k_base")


def _encoding_for_model(model: str) -> tiktoken.Encoding:
    """Resolve the encoding for a model name, with prefix fallback."""
    # Exact match first
    if model in MODEL_ENCODING_MAP:
        return _get_encoding(MODEL_ENCODING_MAP[model])
    # Prefix fallback (e.g. "o3-mini-2025-01-31" -> "o3-mini")
    for prefix, encoding in MODEL_ENCODING_MAP.items():
        if model.startswith(prefix):
            return _get_encoding(encoding)
    # Default: cl100k_base (conservative)
    return _get_encoding("cl100k_base")


# ── Token counter ─────────────────────────────────────────────────────────────


class TokenCounter:
    """
    Accurate token counting with safety margins.

    Always count tokens BEFORE sending to the API to avoid context
    overflow errors at the provider (which cost real money in retries).
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._encoding = _encoding_for_model(model)

    # ── Plain text counting ───────────────────────────────────────────────

    def count_tokens(self, text: str | None) -> int:
        """Count tokens in a text string. None / empty -> 0."""
        if not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token count failed for model {self.model}: {e}")
            # Fallback: rough word-based estimate
            return len(text.split()) * 2

    # ── Message-array counting ────────────────────────────────────────────

    def count_messages_tokens(
        self,
        messages: list[dict],
    ) -> int:
        """
        Count tokens in a chat-format messages array.

        Includes OpenAI's per-message overhead (4 tokens for role +
        content markers) and 3-token priming for the assistant reply.
        Handles multimodal (vision) content blocks.
        """
        total = 0
        for message in messages:
            total += 4  # per-message overhead
            for key, value in message.items():
                if key not in (
                    "role",
                    "content",
                    "name",
                    "tool_calls",
                    "tool_call_id",
                    "function_call",
                ):
                    continue
                if isinstance(value, str):
                    total += self.count_tokens(value)
                elif isinstance(value, list):
                    # Multimodal content (vision): list of {type, text|image_url}
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            total += self.count_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Vision tokens are charged at the provider's
                            # tile-based rate; we approximate as 85 tokens
                            # per image (low-detail tier).
                            total += 85
                elif isinstance(value, dict):
                    # tool_calls / function_call dicts
                    total += self.count_tokens(str(value))
        total += 3  # assistant priming
        return total

    # ── Limit lookups ──────────────────────────────────────────────────────

    def get_context_limit(self, model: str | None = None) -> int:
        """Get context window limit for a model."""
        m = model or self.model
        if m in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[m]
        # Prefix fallback
        for prefix, limit in MODEL_CONTEXT_LIMITS.items():
            if m.startswith(prefix):
                return limit
        return 8_000  # conservative default

    def get_output_limit(self, model: str | None = None) -> int:
        """Get max output tokens for a model."""
        m = model or self.model
        if m in MODEL_OUTPUT_LIMITS:
            return MODEL_OUTPUT_LIMITS[m]
        for prefix, limit in MODEL_OUTPUT_LIMITS.items():
            if m.startswith(prefix):
                return limit
        return 4_096

    # ── Fit-checking ──────────────────────────────────────────────────────

    def check_fits_in_context(
        self,
        messages: list[dict],
        reserved_output_tokens: int = 2_048,
        model: str | None = None,
    ) -> tuple[bool, int, int]:
        """
        Check if messages fit in context window.

        Returns:
            (fits, token_count, available_tokens)
            available_tokens is clamped at 0 (never negative).
        """
        m = model or self.model
        context_limit = self.get_context_limit(m)
        token_count = self.count_messages_tokens(messages)
        available = context_limit - token_count - reserved_output_tokens
        fits = available >= 0

        if not fits:
            logger.warning(
                f"Messages ({token_count} tokens) exceed context limit "
                f"({context_limit}) for model {m}. "
                f"Overflow: {-available} tokens."
            )

        return fits, token_count, max(0, available)

    # ── Truncation ─────────────────────────────────────────────────────────

    def truncate_text_to_tokens(
        self,
        text: str,
        max_tokens: int,
        truncation_marker: str = "\n\n[Content truncated due to length]",
    ) -> str:
        """
        Truncate text to fit within token limit, preserving the start.

        The truncation marker is appended to make truncation visible to
        downstream consumers (especially LLMs, which need to know the
        content was cut).
        """
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        marker_tokens = self.count_tokens(truncation_marker)
        target_tokens = max(max_tokens - marker_tokens, 1)

        encoded = self._encoding.encode(text)
        truncated = self._encoding.decode(encoded[:target_tokens])

        logger.debug(
            f"Truncated text from {current_tokens} to "
            f"~{self.count_tokens(truncated + truncation_marker)} tokens"
        )
        return truncated + truncation_marker

    def truncate_messages_to_fit(
        self,
        messages: list[dict],
        max_tokens: int,
        preserve_system: bool = True,
        preserve_last_n: int = 2,
    ) -> list[dict]:
        """
        Intelligently truncate messages list to fit a token budget.

        Strategy:
            1. Always keep system message(s).
            2. Always keep the last N non-system messages (most recent context).
            3. Drop middle messages (oldest first) until it fits.
            4. If still over, truncate the longest remaining message's content.
        """
        fits, _, _ = self.check_fits_in_context(messages, reserved_output_tokens=max_tokens)
        if fits:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        preserved_end = non_system[-preserve_last_n:] if preserve_last_n else []
        middle = non_system[:-preserve_last_n] if preserve_last_n else non_system

        # Drop middle messages oldest-first until it fits
        while middle:
            result = system_msgs + middle + preserved_end
            fits, _, _ = self.check_fits_in_context(result, reserved_output_tokens=max_tokens)
            if fits:
                return result
            middle = middle[1:]

        # Last resort: truncate each non-system message's content
        result = system_msgs + middle + preserved_end
        for msg in result:
            if msg.get("role") == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = self.truncate_text_to_tokens(content, max_tokens // 2)
        return result


# ── Convenience function ──────────────────────────────────────────────────────


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """One-shot token count convenience function."""
    return TokenCounter(model).count_tokens(text)
