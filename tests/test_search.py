"""
Tests for the search layer (web_search + content_extractor + aggregator).

These are unit tests — no network. We mock out the providers and the
HTTP session so the tests run fast and deterministically in CI.
"""

from __future__ import annotations

import pytest
from src.search.search_aggregator import SearchAggregator
from src.search.web_search import (
    DuckDuckGoProvider,
    SearchResponse,
    SearchResult,
    _is_safe_url,
    _validate_query,
)

# ── URL safety ───────────────────────────────────────────────────────────────


class TestUrlSafety:
    def test_http_url_is_safe(self) -> None:
        assert _is_safe_url("https://example.com/article")

    def test_file_url_is_unsafe(self) -> None:
        assert not _is_safe_url("file:///etc/passwd")

    def test_ftp_url_is_unsafe(self) -> None:
        assert not _is_safe_url("ftp://example.com/x")

    def test_empty_url_is_unsafe(self) -> None:
        assert not _is_safe_url("")

    def test_no_scheme_is_unsafe(self) -> None:
        assert not _is_safe_url("example.com/foo")


# ── Query validation ──────────────────────────────────────────────────────────


class TestQueryValidation:
    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_query("")

    def test_whitespace_query_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_query("   ")

    def test_huge_query_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_query("a" * 2_001)

    def test_normal_query_passes(self) -> None:
        # Should not raise
        _validate_query("latest fusion energy breakthroughs 2026")


# ── DuckDuckGo provider availability ──────────────────────────────────────────


class TestDuckDuckGoProvider:
    def test_name(self) -> None:
        provider = DuckDuckGoProvider(max_results=5)
        assert provider.name == "duckduckgo"

    def test_max_results_propagated(self) -> None:
        provider = DuckDuckGoProvider(max_results=7)
        assert provider.max_results == 7

    def test_is_available_returns_bool(self) -> None:
        provider = DuckDuckGoProvider()
        assert isinstance(provider.is_available, bool)


# ── Aggregator dedup ──────────────────────────────────────────────────────────


class TestAggregatorDedup:
    def setup_method(self) -> None:
        self.agg = SearchAggregator()

    def _make_result(self, url: str, snippet: str = "test snippet content") -> SearchResult:
        return SearchResult(
            title="Test",
            url=url,
            snippet=snippet,
            position=1,
            provider="duckduckgo",
        )

    def test_dedup_removes_exact_url_dupes(self) -> None:
        r1 = self._make_result("https://example.com/a")
        r2 = self._make_result("https://example.com/a")
        deduped = self.agg._deduplicate([r1, r2])
        assert len(deduped) == 1

    def test_dedup_normalizes_www_prefix(self) -> None:
        r1 = self._make_result("https://www.example.com/path")
        r2 = self._make_result("https://example.com/path")
        deduped = self.agg._deduplicate([r1, r2])
        assert len(deduped) == 1

    def test_dedup_normalizes_trailing_slash(self) -> None:
        r1 = self._make_result("https://example.com/path")
        r2 = self._make_result("https://example.com/path/")
        deduped = self.agg._deduplicate([r1, r2])
        assert len(deduped) == 1

    def test_dedup_keeps_different_urls(self) -> None:
        r1 = self._make_result("https://example.com/a")
        r2 = self._make_result("https://example.com/b")
        deduped = self.agg._deduplicate([r1, r2])
        assert len(deduped) == 2

    def test_dedup_drops_high_similarity_snippets(self) -> None:
        r1 = self._make_result(
            "https://example.com/a",
            snippet="deep research pipeline for autonomous reasoning",
        )
        r2 = self._make_result(
            "https://example.com/b",
            snippet="deep research pipeline for autonomous reasoning",
        )
        deduped = self.agg._deduplicate([r1, r2])
        assert len(deduped) == 1


# ── Aggregator relevance scoring ──────────────────────────────────────────────


class TestAggregatorRelevance:
    def setup_method(self) -> None:
        self.agg = SearchAggregator()

    def _make_result(
        self,
        title: str,
        snippet: str,
        domain: str = "example.com",
        position: int = 1,
    ) -> SearchResult:
        return SearchResult(
            title=title,
            url=f"https://{domain}/x",
            snippet=snippet,
            position=position,
            domain=domain,
            provider="duckduckgo",
        )

    def test_score_in_unit_interval(self) -> None:
        result = self._make_result("Title", "Snippet with query terms")
        score = self.agg._compute_relevance(result, {"query", "terms"})
        assert 0.0 <= score <= 1.0

    def test_trusted_domain_gets_boost(self) -> None:
        wiki = self._make_result(
            "Fusion energy",
            "Fusion energy article",
            domain="en.wikipedia.org",
        )
        random = self._make_result(
            "Fusion energy",
            "Fusion energy article",
            domain="random-blog.example",
        )
        score_wiki = self.agg._compute_relevance(wiki, {"fusion", "energy"})
        score_random = self.agg._compute_relevance(random, {"fusion", "energy"})
        assert score_wiki >= score_random

    def test_low_quality_domain_gets_penalty(self) -> None:
        good = self._make_result(
            "Fusion energy",
            "Fusion energy breakthrough",
            domain="nature.com",
        )
        bad = self._make_result(
            "Fusion energy",
            "Fusion energy breakthrough",
            domain="pinterest.com",
        )
        assert self.agg._compute_relevance(
            good, {"fusion", "energy"}
        ) > self.agg._compute_relevance(bad, {"fusion", "energy"})

    def test_earlier_position_scores_higher(self) -> None:
        early = self._make_result("T", "snippet", position=1)
        late = self._make_result("T", "snippet", position=10)
        assert self.agg._compute_relevance(early, set()) > self.agg._compute_relevance(late, set())

    def test_query_term_coverage_matters(self) -> None:
        full = self._make_result("alpha beta", "alpha beta gamma")
        partial = self._make_result("alpha", "alpha")
        assert self.agg._compute_relevance(
            full, {"alpha", "beta", "gamma"}
        ) > self.agg._compute_relevance(partial, {"alpha", "beta", "gamma"})


# ── Aggregator context building ───────────────────────────────────────────────


class TestAggregatorContext:
    def setup_method(self) -> None:
        self.agg = SearchAggregator()

    def test_empty_results_returns_fallback(self) -> None:
        ctx = self.agg.build_context_string([], max_total_tokens=1_000)
        assert "No search results" in ctx

    def test_context_includes_source_title(self) -> None:
        result = SearchResult(
            title="My Source",
            url="https://example.com/x",
            snippet="snippet text",
            domain="example.com",
            provider="duckduckgo",
            position=1,
            relevance_score=0.5,
        )
        from src.search.search_aggregator import AggregatedResult

        agg = AggregatedResult(search_result=result, relevance_score=0.5)
        ctx = self.agg.build_context_string([agg], max_total_tokens=1_000)
        assert "My Source" in ctx


# ── SearchResponse dataclass ──────────────────────────────────────────────────


class TestSearchResponse:
    def test_has_results_true_when_results_present(self) -> None:
        r = SearchResult(
            title="t",
            url="https://x.com",
            snippet="s",
            position=1,
            provider="test",
        )
        resp = SearchResponse(query="q", results=[r], provider="test")
        assert resp.has_results

    def test_has_results_false_when_results_empty(self) -> None:
        resp = SearchResponse(query="q", results=[], provider="test")
        assert not resp.has_results

    def test_has_results_false_when_error_set(self) -> None:
        resp = SearchResponse(
            query="q",
            results=[],
            provider="test",
            error="boom",
        )
        assert not resp.has_results
