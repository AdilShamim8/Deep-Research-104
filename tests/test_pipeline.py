"""Integration tests for the full pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.base_model import ModelResponse


def make_research_model(answer: str = "Fusion energy is advancing."):
    model = MagicMock()
    model.model_name = "mock-research-model"
    model.get_context_limit.return_value = 128_000

    async def mock_generate(messages, config=None):
        content = ""
        if messages:
            last_content = messages[-1].get("content", "")
            if "numbered" in last_content.lower() or "queries" in last_content.lower():
                content = "1. fusion energy 2024\n2. nuclear fusion breakthrough\n3. ITER progress"
            elif "gaps" in last_content.lower() or "gap" in last_content.lower():
                content = (
                    "<covered>\n- Basic facts\n</covered>\n"
                    "<gaps>\n- Recent developments\n</gaps>\n"
                    "<follow_up_queries>\n1. latest fusion news\n</follow_up_queries>\n"
                    "<confidence>high</confidence>"
                )
            else:
                content = answer

        return ModelResponse(
            content=content,
            model="mock-research-model",
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
        )

    model.generate = mock_generate
    return model


@pytest.mark.asyncio
class TestDeepResearchPipeline:

    async def test_pipeline_runs(self):
        """Test that pipeline completes without errors."""
        from src.pipeline.deep_research import DeepResearchPipeline

        model = make_research_model()

        with patch(
            "src.pipeline.deep_research.search_engine"
        ) as mock_search, patch(
            "src.pipeline.deep_research.extractor"
        ) as mock_extractor:

            # Mock search results
            from src.search.web_search import SearchResponse, SearchResult
            mock_result = SearchResult(
                title="Fusion Energy Article",
                url="https://example.com/fusion",
                snippet="Fusion energy is making progress.",
                position=1,
                provider="mock",
            )
            mock_search_response = SearchResponse(
                query="fusion energy",
                results=[mock_result],
                provider="mock",
                total_results=1,
            )
            mock_search.multi_search = AsyncMock(
                return_value=[mock_search_response]
            )

            # Mock content extraction
            from src.search.content_extractor import ExtractedContent
            mock_content = ExtractedContent(
                url="https://example.com/fusion",
                title="Fusion Energy Article",
                text="Fusion energy research is advancing rapidly. "
                     "ITER project is making significant progress.",
                success=True,
                extraction_method="mock",
            )
            mock_extractor.extract_many = AsyncMock(
                return_value=[mock_content]
            )

            pipeline = DeepResearchPipeline(
                model=model,
                max_iterations=1,
                queries_per_iteration=2,
            )
            report = await pipeline.research(
                "What is fusion energy?",
                fast_mode=True,
            )

            assert report.question == "What is fusion energy?"
            assert report.answer
            assert len(report.answer) > 0
            assert report.total_llm_calls > 0

    async def test_pipeline_generates_queries(self):
        """Test query generation step."""
        from src.pipeline.deep_research import DeepResearchPipeline

        model = make_research_model()
        pipeline = DeepResearchPipeline(model=model, max_iterations=1)

        queries = await pipeline._generate_queries(
            "What is fusion energy?", n=3
        )

        assert isinstance(queries, list)
        assert len(queries) >= 1

    async def test_pipeline_gap_analysis(self):
        """Test gap analysis step."""
        from src.pipeline.deep_research import DeepResearchPipeline

        model = make_research_model()
        pipeline = DeepResearchPipeline(model=model)

        gaps, confidence = await pipeline._analyze_gaps(
            "What is fusion energy?",
            "Fusion uses hydrogen isotopes to produce energy.",
        )

        assert isinstance(gaps, list)
        assert confidence in ("low", "medium", "high")

    async def test_report_to_markdown(self):
        """Test report markdown generation."""
        from src.pipeline.deep_research import ResearchReport

        report = ResearchReport(
            question="Test question?",
            answer="Test answer.",
            sources=[],
            iterations=[],
            confidence="high",
            total_sources_found=5,
            total_search_time_ms=1000.0,
            total_llm_calls=3,
            total_tokens_used=500,
            synthesis_model="test-model",
        )

        md = report.to_markdown()
        assert "# Research Report" in md
        assert "Test question?" in md
        assert "Test answer." in md
        assert "high" in md


@pytest.mark.asyncio
class TestMetaCoT:

    async def test_meta_cot_parses_actions(self):
        """Test action tag parsing."""
        from src.pipeline.meta_cot import MetaCoT

        model = MagicMock()
        model.model_name = "mock"
        model.get_context_limit.return_value = 128_000
        meta_cot = MetaCoT(model)

        text = (
            "<think>I need to search for this.</think>\n"
            "<search>fusion energy 2024</search>\n"
            "<think>Based on results...</think>\n"
            "<answer>Fusion is advancing.</answer>"
        )
        actions = meta_cot._parse_actions(text)

        assert len(actions) == 4
        assert actions[0].action_type == "think"
        assert actions[1].action_type == "search"
        assert actions[1].content == "fusion energy 2024"
        assert actions[3].action_type == "answer"

    async def test_meta_cot_runs_to_answer(self):
        """Test that Meta-CoT terminates on answer tag."""
        from src.pipeline.meta_cot import MetaCoT

        call_count = [0]

        async def mock_generate(messages, config=None):
            call_count[0] += 1
            if call_count[0] == 1:
                content = (
                    "<think>Let me think about this.</think>\n"
                    "<search>test query</search>"
                )
            else:
                content = (
                    "<think>I now have enough info.</think>\n"
                    "<answer>The answer is 42.</answer>"
                )
            return ModelResponse(
                content=content,
                model="mock",
                total_tokens=50,
            )

        model = MagicMock()
        model.model_name = "mock"
        model.get_context_limit.return_value = 128_000
        model.generate = mock_generate

        with patch(
            "src.pipeline.meta_cot.search_engine"
        ) as mock_search:
            from src.search.web_search import SearchResponse
            mock_search.search = AsyncMock(
                return_value=SearchResponse(
                    query="test",
                    results=[],
                    provider="mock",
                )
            )

            meta_cot = MetaCoT(
                model=model,
                max_searches=5,
                max_iterations=5,
            )
            result = await meta_cot.reason("What is 6 x 7?")

            assert result.answer == "The answer is 42."
            assert result.total_searches == 1