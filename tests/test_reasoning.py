"""Tests for reasoning modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.base_model import ModelResponse, GenerationConfig


def make_mock_model(response_text: str, reasoning: str = "") -> MagicMock:
    """Create a mock model that returns given response."""
    model = MagicMock()
    model.model_name = "mock-model"
    model.get_context_limit.return_value = 128_000

    mock_response = ModelResponse(
        content=response_text,
        reasoning_content=reasoning or None,
        model="mock-model",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    model.generate = AsyncMock(return_value=mock_response)
    return model


@pytest.mark.asyncio
class TestChainOfThought:

    async def test_zero_shot_generate(self):
        from src.reasoning.chain_of_thought import ChainOfThought

        model = make_mock_model(
            "<reasoning>Step 1: Think.</reasoning>"
            "<answer>42</answer>"
        )
        cot = ChainOfThought(model)
        result = await cot.generate("What is 6x7?", mode="zero_shot")

        assert result.answer == "42"
        model.generate.assert_called_once()

    async def test_structured_generate(self):
        from src.reasoning.chain_of_thought import ChainOfThought

        model = make_mock_model(
            "<reasoning>The answer is computed as follows.</reasoning>"
            "<answer>The result is 42.</answer>"
        )
        cot = ChainOfThought(model)
        result = await cot.generate("What is 6x7?", mode="structured")

        assert "42" in result.answer
        assert "computed" in result.reasoning

    async def test_self_consistency_majority_vote(self):
        from src.reasoning.chain_of_thought import ChainOfThought

        responses = [
            "<answer>Paris</answer>",
            "<answer>Paris</answer>",
            "<answer>London</answer>",
            "<answer>Paris</answer>",
            "<answer>Berlin</answer>",
        ]
        call_count = 0

        async def mock_generate(messages, config=None):
            nonlocal call_count
            resp = ModelResponse(
                content=responses[call_count % len(responses)],
                model="mock-model",
                total_tokens=50,
            )
            call_count += 1
            return resp

        model = MagicMock()
        model.model_name = "mock-model"
        model.get_context_limit.return_value = 128_000
        model.generate = mock_generate

        cot = ChainOfThought(model)
        result = await cot.self_consistency(
            "What is the capital of France?",
            n_samples=5,
        )

        assert "Paris" in result.answer
        assert result.supporting_votes == 3
        assert result.confidence == pytest.approx(0.6)

    async def test_cot_with_reasoning_content(self):
        """Test that model reasoning_content takes priority."""
        from src.reasoning.chain_of_thought import ChainOfThought

        model = make_mock_model(
            response_text="The answer is 42",
            reasoning=("I need to multiply 6 by 7. "
                        "6 times 7 equals 42."),
        )
        cot = ChainOfThought(model)
        result = await cot.generate("6 x 7?")

        assert result.reasoning == ("I need to multiply 6 by 7. "
                                    "6 times 7 equals 42.")


@pytest.mark.asyncio
class TestParallelSampler:

    async def test_sample_best_of_n(self):
        from src.reasoning.parallel_sampling import ParallelSampler

        responses = [
            "Short answer.",
            "Medium length answer with some detail about the topic.",
            "Longer answer with step-by-step reasoning about why the "
            "answer is correct, therefore we conclude X.",
        ]
        idx = 0

        async def mock_generate(messages, config=None):
            nonlocal idx
            r = ModelResponse(
                content=responses[idx % len(responses)],
                model="mock",
                total_tokens=50,
            )
            idx += 1
            return r

        model = MagicMock()
        model.model_name = "mock"
        model.get_context_limit.return_value = 128_000
        model.generate = mock_generate

        sampler = ParallelSampler(model)
        result = await sampler.sample_best_of_n(
            messages=[{"role": "user", "content": "test"}],
            n=3,
            selection="score",
        )

        assert result.n_generated == 3
        assert result.n_successful == 3
        assert result.best_content  # Non-empty
        assert len(result.all_samples) == 3

    async def test_text_similarity(self):
        from src.reasoning.parallel_sampling import ParallelSampler

        sim = ParallelSampler._text_similarity(
            "The cat sat on the mat",
            "The cat sat on the mat",
        )
        assert sim == pytest.approx(1.0)

        sim2 = ParallelSampler._text_similarity(
            "The cat sat on the mat",
            "Completely different words here",
        )
        assert sim2 < 0.2


@pytest.mark.asyncio
class TestTreeOfThoughts:

    async def test_parse_thoughts(self):
        from src.reasoning.tree_of_thoughts import TreeOfThoughts

        text = (
            "STEP: First, analyze the problem carefully.\n"
            "STEP: Then, consider all possible approaches.\n"
            "STEP: Finally, select the best solution."
        )
        thoughts = TreeOfThoughts._parse_thoughts(text)
        assert len(thoughts) == 3
        assert "analyze" in thoughts[0]

    async def test_solve_mock(self):
        from src.reasoning.tree_of_thoughts import (
            TreeOfThoughts, SearchStrategy
        )

        call_num = [0]

        async def smart_mock(messages, config=None):
            call_num[0] += 1
            content = messages[0]["content"]

            if "NEXT STEPS" in content or "Generate" in content:
                return ModelResponse(
                    content=(
                        "STEP: Analyze the numbers.\n"
                        "STEP: Apply arithmetic.\n"
                        "STEP: Verify result."
                    ),
                    model="mock",
                    total_tokens=50,
                )
            elif "0.0 to 1.0" in content:
                return ModelResponse(
                    content="0.8",
                    model="mock",
                    total_tokens=5,
                )
            elif "YES or NO" in content:
                return ModelResponse(
                    content="YES",
                    model="mock",
                    total_tokens=3,
                )
            else:
                return ModelResponse(
                    content="The answer is 42.",
                    model="mock",
                    total_tokens=20,
                )

        model = MagicMock()
        model.model_name = "mock"
        model.get_context_limit.return_value = 128_000
        model.generate = smart_mock

        tot = TreeOfThoughts(
            model=model,
            branching_factor=2,
            max_depth=2,
            beam_width=1,
            search_strategy=SearchStrategy.BEAM,
        )
        result = await tot.solve("What is 6 x 7?")

        assert result.answer
        assert result.total_nodes_explored > 0


@pytest.mark.asyncio
class TestVerifier:

    async def test_orm_parse_response(self):
        from src.reasoning.verifier import OutcomeRewardModel

        model = make_mock_model(
            "SCORE: 0.9\nCORRECT: YES\nFEEDBACK: Excellent answer."
        )
        orm = OutcomeRewardModel(model)
        result = await orm.verify(
            question="What is 2+2?",
            answer="4",
        )

        assert result.score == pytest.approx(0.9)
        assert result.is_correct is True
        assert "Excellent" in result.feedback

    async def test_rule_based_exact_match(self):
        from src.reasoning.verifier import RuleBasedVerifier

        verifier = RuleBasedVerifier()
        result = await verifier.verify(
            question="What is 2+2?",
            answer="4",
            reference="4",
        )
        assert result.is_correct is True
        assert result.score == 1.0

    async def test_rule_based_numeric_tolerance(self):
        from src.reasoning.verifier import RuleBasedVerifier

        verifier = RuleBasedVerifier(numeric_tolerance=0.01)
        result = await verifier.verify(
            question="What is pi?",
            answer="3.14159",
            reference="3.14159",
        )
        assert result.is_correct is True

    async def test_rule_based_wrong_answer(self):
        from src.reasoning.verifier import RuleBasedVerifier

        verifier = RuleBasedVerifier()
        result = await verifier.verify(
            question="Capital of France?",
            answer="London",
            reference="Paris",
        )
        assert result.is_correct is False
        assert result.score == 0.0