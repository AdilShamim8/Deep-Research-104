"""Tests for model providers."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.models.base_model import GenerationConfig, ModelResponse
from src.utils.token_counter import TokenCounter, MODEL_CONTEXT_LIMITS


class TestTokenCounter:

    def test_count_tokens_basic(self):
        counter = TokenCounter("gpt-4o")
        count = counter.count_tokens("Hello world")
        assert count > 0
        assert count < 10

    def test_count_tokens_empty(self):
        counter = TokenCounter("gpt-4o")
        assert counter.count_tokens("") == 0

    def test_count_messages_tokens(self):
        counter = TokenCounter("gpt-4o")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        count = counter.count_messages_tokens(messages)
        assert count > 0

    def test_check_fits_in_context_fits(self):
        counter = TokenCounter("gpt-4o")
        messages = [{"role": "user", "content": "Hi"}]
        fits, count, available = counter.check_fits_in_context(
            messages, reserved_output_tokens=100
        )
        assert fits is True
        assert count > 0
        assert available > 0

    def test_check_fits_in_context_overflow(self):
        counter = TokenCounter("gpt-4o")
        # Create a very long message
        long_text = "word " * 150_000  # Way over any limit
        messages = [{"role": "user", "content": long_text}]
        fits, count, available = counter.check_fits_in_context(
            messages, reserved_output_tokens=100
        )
        assert fits is False
        assert available == 0

    def test_truncate_text_to_tokens(self):
        counter = TokenCounter("gpt-4o")
        long_text = "This is a test sentence. " * 1000
        truncated = counter.truncate_text_to_tokens(long_text, max_tokens=50)
        assert counter.count_tokens(truncated) <= 60  # With marker
        assert "[Content truncated" in truncated

    def test_truncate_short_text_unchanged(self):
        counter = TokenCounter("gpt-4o")
        short_text = "Short text."
        result = counter.truncate_text_to_tokens(short_text, max_tokens=100)
        assert result == short_text

    def test_context_limit_known_model(self):
        counter = TokenCounter("o3")
        assert counter.get_context_limit() == 200_000

    def test_context_limit_unknown_model(self):
        counter = TokenCounter("unknown-model-xyz")
        assert counter.get_context_limit() == 8_000

    def test_truncate_messages_to_fit(self):
        counter = TokenCounter("gpt-4o")
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Latest question"},
        ]
        trimmed = counter.truncate_messages_to_fit(
            messages,
            max_tokens=200,
            preserve_system=True,
            preserve_last_n=2,
        )
        # System message should be preserved
        assert any(m["role"] == "system" for m in trimmed)
        # Last message should be preserved
        assert trimmed[-1]["content"] == "Latest question"


class TestModelResponse:

    def test_model_response_defaults(self):
        resp = ModelResponse(content="Hello", model="test-model")
        assert resp.content == "Hello"
        assert resp.reasoning_content is None
        assert resp.total_cost_estimate == 0.0
        assert resp.alternatives == []

    def test_model_response_with_alternatives(self):
        resp = ModelResponse(
            content="Main answer",
            model="gpt-4o",
            alternatives=["Alt 1", "Alt 2"],
        )
        assert len(resp.alternatives) == 2


@pytest.mark.asyncio
class TestOpenAIModel:

    async def test_generate_mock(self):
        """Test OpenAI model with mocked API call."""
        from src.models.openai_models import OpenAIModel

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel("gpt-4o")
            model._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            response = await model.generate(
                messages=[{"role": "user", "content": "Hi"}],
                config=GenerationConfig(max_tokens=100),
            )

            assert response.content == "Test answer"
            assert response.total_tokens == 15

    async def test_o_series_system_message_conversion(self):
        """Test that system messages are converted for o-series models."""
        from src.models.openai_models import OpenAIModel

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel("o3-mini")
            messages = [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ]
            prepared = model._prepare_messages(messages)

            # System message should be merged into user message
            assert not any(m["role"] == "system" for m in prepared)
            assert any(
                "Be helpful." in m["content"]
                for m in prepared if m["role"] == "user"
            )

    async def test_validate_messages_overflow(self):
        """Test that token overflow raises ValueError."""
        from src.models.openai_models import OpenAIModel

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel("gpt-4o")
            huge_messages = [
                {"role": "user", "content": "word " * 200_000}
            ]
            with pytest.raises(ValueError, match="exceed context limit"):
                model.validate_messages(huge_messages)