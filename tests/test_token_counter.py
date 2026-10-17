"""
Unit tests for src.utils.token_counter.

Covers:
- Plain-text token counting (deterministic for ASCII, approximate for CJK)
- Message-array counting with per-message overhead
- Multimodal content blocks (image_url → 85 tokens approximation)
- Context-limit + output-limit lookups with prefix fallback
- Truncation (preserves start, appends marker)
- Defensive behavior (None, empty, non-ASCII fallback)
"""

from __future__ import annotations

from src.utils.token_counter import (
    MODEL_CONTEXT_LIMITS,
    MODEL_OUTPUT_LIMITS,
    TokenCounter,
    count_tokens,
)

# ── Plain text counting ──────────────────────────────────────────────────────


class TestCountTokens:
    def test_empty_returns_zero(self) -> None:
        assert TokenCounter().count_tokens("") == 0

    def test_none_returns_zero(self) -> None:
        assert TokenCounter().count_tokens(None) == 0  # type: ignore

    def test_ascii_text_has_positive_count(self) -> None:
        assert TokenCounter().count_tokens("hello world") > 0

    def test_longer_text_has_more_tokens(self) -> None:
        c = TokenCounter()
        assert c.count_tokens("a") < c.count_tokens("a " * 100)

    def test_count_tokens_helper(self) -> None:
        assert count_tokens("hello", "gpt-4o") > 0


# ── Message-array counting ──────────────────────────────────────────────────────


class TestCountMessages:
    def test_single_user_message_includes_overhead(self) -> None:
        c = TokenCounter()
        msgs = [{"role": "user", "content": "hello"}]
        text_tokens = c.count_tokens("hello")
        msg_tokens = c.count_messages_tokens(msgs)
        # 4 tokens overhead per message + 3 priming
        assert msg_tokens >= text_tokens + 4 + 3

    def test_more_messages_more_tokens(self) -> None:
        c = TokenCounter()
        few = [{"role": "user", "content": "x"}]
        many = [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]
        assert c.count_messages_tokens(many) > c.count_messages_tokens(few)

    def test_multimodal_image_counts_85_tokens(self) -> None:
        c = TokenCounter()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "http://x"}},
                ],
            }
        ]
        text_only = [{"role": "user", "content": "describe this"}]
        assert c.count_messages_tokens(msgs) - c.count_messages_tokens(text_only) == 85


# ── Context / output limits ───────────────────────────────────────────────────


class TestLimits:
    def test_known_model_context_limit(self) -> None:
        c = TokenCounter("gpt-4o")
        assert c.get_context_limit() == MODEL_CONTEXT_LIMITS["gpt-4o"]

    def test_known_model_output_limit(self) -> None:
        c = TokenCounter("gpt-4o")
        assert c.get_output_limit() == MODEL_OUTPUT_LIMITS["gpt-4o"]

    def test_unknown_model_falls_back(self) -> None:
        c = TokenCounter("totally-unknown-model-xyz")
        assert c.get_context_limit() > 0
        assert c.get_output_limit() > 0

    def test_prefix_fallback_for_versioned_model(self) -> None:
        c = TokenCounter("o3-mini-2025-01-31")
        assert c.get_context_limit() == MODEL_CONTEXT_LIMITS["o3-mini"]


# ── Fit checking ────────────────────────────────────────────────────────────────


class TestFitsInContext:
    def test_small_messages_fit(self) -> None:
        c = TokenCounter("gpt-4o")
        msgs = [{"role": "user", "content": "hi"}]
        fits, count, available = c.check_fits_in_context(msgs)
        assert fits
        assert count > 0
        assert available > 0

    def test_oversized_messages_do_not_fit(self) -> None:
        c = TokenCounter("gpt-4o")
        # 10M chars ≈ 1.25M tokens — way over any model's context limit
        huge = "x" * 10_000_000
        msgs = [{"role": "user", "content": huge}]
        fits, _count, available = c.check_fits_in_context(msgs)
        assert not fits
        assert available == 0  # clamped


# ── Truncation ──────────────────────────────────────────────────────────────────


class TestTruncation:
    def test_short_text_unchanged(self) -> None:
        c = TokenCounter()
        text = "short"
        assert c.truncate_text_to_tokens(text, 1_000) == text

    def test_long_text_gets_marker(self) -> None:
        c = TokenCounter()
        text = "word " * 10_000
        truncated = c.truncate_text_to_tokens(text, max_tokens=500)
        assert "[Content truncated due to length]" in truncated

    def test_truncated_text_within_budget(self) -> None:
        c = TokenCounter("gpt-4o")
        text = "word " * 10_000
        truncated = c.truncate_text_to_tokens(text, max_tokens=500)
        assert c.count_tokens(truncated) <= 500


# ── Message truncation ────────────────────────────────────────────────────────────


class TestTruncateMessages:
    def test_small_messages_unchanged(self) -> None:
        c = TokenCounter("gpt-4o")
        msgs = [{"role": "user", "content": "hi"}]
        result = c.truncate_messages_to_fit(msgs, max_tokens=2_000)
        assert result == msgs

    def test_system_message_preserved(self) -> None:
        c = TokenCounter("gpt-4o")
        msgs = [
            {"role": "system", "content": "important instructions"},
            {"role": "user", "content": "x" * 100_000},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "latest question"},
        ]
        result = c.truncate_messages_to_fit(msgs, max_tokens=500, preserve_last_n=2)
        roles = [m["role"] for m in result]
        assert "system" in roles
