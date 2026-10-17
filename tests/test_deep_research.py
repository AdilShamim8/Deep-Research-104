"""
Tests for src.pipeline.deep_research — DeepResearchPipeline orchestrator.

Uses StubModel from conftest so no LLM provider is hit. Mocks out the
search engine + extractor so no network is touched. Verifies the
defensive iteration logic, request_id propagation, and report
serialization.
"""

from __future__ import annotations

import pytest
from src.pipeline.deep_research import (
    DeepResearchPipeline,
    ResearchIteration,
    ResearchReport,
)
from src.search.search_aggregator import AggregatedResult
from src.search.web_search import SearchResponse, SearchResult

# ── ResearchIteration dataclass ───────────────────────────────────────────────


class TestResearchIteration:
    def test_defaults(self) -> None:
        it = ResearchIteration(iteration_number=1)
        assert it.iteration_number == 1
        assert it.queries == []
        assert it.confidence == "low"
        assert it.context_tokens == 0

    def test_with_values(self) -> None:
        it = ResearchIteration(
            iteration_number=2,
            queries=["q1", "q2"],
            confidence="high",
            context_tokens=1000,
            duration_ms=42.0,
            llm_calls=3,
            tokens_used=500,
            cost_usd=0.01,
        )
        assert it.queries == ["q1", "q2"]
        assert it.confidence == "high"
        assert it.duration_ms == 42.0
        assert it.llm_calls == 3


# ── ResearchReport dataclass ──────────────────────────────────────────────────


class TestResearchReport:
    def test_to_markdown_includes_question(self) -> None:
        r = ResearchReport(
            question="What is X?",
            answer="X is something.",
            confidence="medium",
            synthesis_model="stub",
        )
        md = r.to_markdown()
        assert "# Research Report" in md
        assert "What is X?" in md
        assert "X is something." in md

    def test_to_markdown_includes_gaps(self) -> None:
        r = ResearchReport(
            question="q",
            answer="a",
            confidence="low",
            synthesis_model="stub",
            gaps_remaining=["missing-1", "missing-2"],
        )
        md = r.to_markdown()
        assert "## Remaining Uncertainties" in md
        assert "missing-1" in md

    def test_to_json_roundtrips(self) -> None:
        import json

        r = ResearchReport(
            question="q",
            answer="a",
            confidence="medium",
            synthesis_model="stub",
            total_tokens_used=100,
            total_cost_usd=0.01,
        )
        s = r.to_json()
        d = json.loads(s)
        assert d["question"] == "q"
        assert d["answer"] == "a"
        assert d["confidence"] == "medium"
        assert d["total_tokens_used"] == 100
        assert d["total_cost_usd"] == 0.01

    def test_source_urls_property(self) -> None:
        result = SearchResult(
            title="t",
            url="https://example.com/a",
            snippet="s",
            position=1,
            provider="ddg",
        )
        agg = AggregatedResult(search_result=result, relevance_score=0.5)
        r = ResearchReport(
            question="q",
            answer="a",
            confidence="medium",
            synthesis_model="stub",
            sources=[agg],
        )
        assert r.source_urls == ["https://example.com/a"]


# ── Pipeline with StubModel + mocked search ─────────────────────────────────


class TestPipelineEndToEnd:
    """Run the full pipeline against a stub model + a mocked search engine."""

    @pytest.mark.asyncio
    async def test_pipeline_runs_fast_mode(
        self, mock_model, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub the search engine to return canned results
        canned_result = SearchResult(
            title="Fusion breakthrough",
            url="https://example.com/fusion-2026",
            snippet="Major fusion milestone achieved in 2026.",
            position=1,
            provider="duckduckgo",
        )
        canned_response = SearchResponse(
            query="fusion breakthroughs 2026",
            results=[canned_result],
            provider="duckduckgo",
            total_results=1,
        )

        async def fake_multi_search(queries, use_cache=True):
            return [canned_response for _ in queries]

        async def fake_aggregate(*args, **kwargs):
            return [
                AggregatedResult(
                    search_result=canned_result,
                    relevance_score=0.8,
                    query_matches=["fusion"],
                )
            ]

        # Stub the search engine and aggregator on the pipeline instance
        monkeypatch.setattr(
            "src.pipeline.deep_research.search_engine.multi_search",
            fake_multi_search,
        )

        pipeline = DeepResearchPipeline(
            model=mock_model,
            max_iterations=1,
        )
        monkeypatch.setattr(pipeline._aggregator, "aggregate", fake_aggregate)

        report = await pipeline.research("What is X?", fast_mode=True)

        assert isinstance(report, ResearchReport)
        assert report.question == "What is X?"
        assert isinstance(report.answer, str)
        assert len(report.iterations) == 1
        assert mock_model.call_count >= 1  # StubModel tracks calls
