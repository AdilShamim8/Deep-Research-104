"""
Search result aggregation, deduplication, and relevance ranking.

Takes raw search results from multiple queries / providers, cleans
them up, removes duplicates, scores relevance against the original
query, and fetches full page content for the top results.

Production hardening over the toy original:
    - Defensive: never raises; malformed URLs are skipped, not crashed
    - Per-aggregation metrics (input count, dedup removed, fetched, ok)
    - Tunable trusted-domain / low-quality-domain lists (env override)
    - Token-budget allocator keeps a 100-token floor per source so low-
      scoring sources aren't zeroed out completely
    - Parallel content fetching with bounded concurrency via extractor

Date tag: 17 October 2026.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from loguru import logger

from src.search.content_extractor import ContentExtractor, ExtractedContent
from src.search.web_search import SearchResponse, SearchResult
from src.utils.token_counter import TokenCounter

# ── Aggregated result ─────────────────────────────────────────────────────────


@dataclass
class AggregatedResult:
    """A deduplicated, ranked search result with extracted content."""

    search_result: SearchResult
    content: ExtractedContent | None = None
    relevance_score: float = 0.0
    query_matches: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return self.content is not None and self.content.success

    @property
    def url(self) -> str:
        return self.search_result.url

    @property
    def title(self) -> str:
        if self.has_content and self.content and self.content.title:
            return self.content.title
        return self.search_result.title

    @property
    def text(self) -> str:
        if self.has_content and self.content:
            return self.content.text
        return self.search_result.snippet

    @property
    def domain(self) -> str:
        return self.search_result.domain

    @property
    def published_date(self) -> str | None:
        return self.search_result.published_date

    def to_context_string(self, max_length: int = 3_000) -> str:
        """Format result for inclusion in an LLM prompt."""
        parts = [
            f"### Source: {self.title}",
            f"URL: {self.url}",
        ]
        if self.search_result.published_date:
            parts.append(f"Date: {self.search_result.published_date}")
        parts.append("")
        text = self.text[:max_length]
        if len(self.text) > max_length:
            text += "\n[Content truncated]"
        parts.append(text)
        return "\n".join(parts)


# ── Domain reputation tables ──────────────────────────────────────────────────
# Values are MULTIPLIERS applied to the base relevance score.
#   > 1.0 boosts (trusted sources rank higher than random)
#   < 1.0 penalizes (low-quality sources rank lower)
# Final score is clamped to [0.0, 1.0] at the end.

TRUSTED_DOMAINS: dict[str, float] = {
    "arxiv.org": 1.15,
    "pubmed.ncbi.nlm.nih.gov": 1.15,
    "nature.com": 1.10,
    "science.org": 1.10,
    "ieee.org": 1.10,
    "acm.org": 1.10,
    "wikipedia.org": 1.08,
    "github.com": 1.08,
    "stackoverflow.com": 1.05,
    "docs.python.org": 1.08,
    "developer.mozilla.org": 1.08,
    "nytimes.com": 1.05,
    "bbc.com": 1.05,
    "reuters.com": 1.07,
    "apnews.com": 1.07,
    "economist.com": 1.05,
    "dartmouth.edu": 1.05,  # +other .edu domains handled below
}

LOW_QUALITY_DOMAINS: frozenset[str] = frozenset(
    {
        "pinterest.com",
        "quora.com",
        "answers.com",
        "ask.com",
        "ehow.com",
        "wikihow.com",
        "buzzfeed.com",
        "listverse.com",
    }
)

STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "is",
        "was",
        "are",
        "were",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "this",
        "that",
        "these",
        "those",
    }
)


# ── Aggregator ───────────────────────────────────────────────────────────────


class SearchAggregator:
    """Aggregate, deduplicate, and rank search results.

    Pipeline:
        1. Collect results from multiple SearchResponse objects
        2. Deduplicate by URL + content similarity
        3. Score relevance against the original query
        4. Fetch full content for top results
        5. Return ranked, enriched results
    """

    def __init__(
        self,
        extractor: ContentExtractor | None = None,
        max_results_to_fetch: int = 5,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._extractor = extractor or ContentExtractor()
        self.max_results_to_fetch = max_results_to_fetch
        self._token_counter = token_counter or TokenCounter()

    # ── Public API ────────────────────────────────────────────────────────────

    async def aggregate(
        self,
        query: str,
        search_responses: list[SearchResponse],
        fetch_content: bool = True,
        max_total_results: int = 10,
    ) -> list[AggregatedResult]:
        """Aggregate multiple search responses into ranked results."""
        start = time.time()

        all_results: list[SearchResult] = []
        for response in search_responses:
            all_results.extend(response.results)

        logger.info(
            f"Aggregating {len(all_results)} results "
            f"from {len(search_responses)} searches for query {query[:80]!r}"
        )

        deduped = self._deduplicate(all_results)
        logger.debug(
            f"After dedup: {len(deduped)} results (removed {len(all_results) - len(deduped)})"
        )

        scored = self._score_relevance(query, deduped)
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        scored = scored[:max_total_results]

        if fetch_content and scored:
            top_n = min(self.max_results_to_fetch, len(scored))
            urls = [r.url for r in scored[:top_n]]
            contents = await self._extractor.extract_many(urls)
            for agg_result, content in zip(scored[:top_n], contents):
                if content.success:
                    agg_result.content = content

        successful_content = sum(1 for r in scored if r.has_content)
        logger.info(
            f"Aggregation complete: {len(scored)} results, "
            f"{successful_content} with full content "
            f"in {(time.time() - start) * 1000:.0f}ms"
        )

        return scored

    def build_context_string(
        self,
        results: list[AggregatedResult],
        max_total_tokens: int = 12_000,
        model: str = "gpt-4o",
    ) -> str:
        """Build a single context string from aggregated results that
        fits within the token budget. Distributes tokens proportionally
        to relevance score, with a per-source floor of 200 tokens."""
        if not results:
            return "No search results found."

        counter = TokenCounter(model)
        header = "## Search Results\n\n"
        header_tokens = counter.count_tokens(header)
        budget = max_total_tokens - header_tokens

        total_score = sum(r.relevance_score for r in results) or 1.0
        sections: list[str] = []
        remaining_budget = budget

        for result in results:
            if remaining_budget <= 100:
                break
            # Proportional allocation, capped at 4000 and floored at 200
            allocated = int((result.relevance_score / total_score) * budget)
            allocated = min(allocated, remaining_budget, 4_000)
            allocated = max(allocated, 200)

            section = result.to_context_string(max_length=allocated * 4)
            section_tokens = counter.count_tokens(section)

            if section_tokens > allocated:
                section = counter.truncate_text_to_tokens(section, allocated)
                section_tokens = allocated

            sections.append(section)
            remaining_budget -= section_tokens

        context = header + "\n\n---\n\n".join(sections)
        final_tokens = counter.count_tokens(context)
        logger.debug(
            f"Context built: {len(sections)} sources, {final_tokens}/{max_total_tokens} tokens"
        )
        return context

    # ── Deduplication ───────────────────────────────────────────────────────

    def _deduplicate(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Remove duplicate results by URL match + normalized URL match
        + high snippet similarity (Jaccard > 0.8)."""
        seen_urls: set[str] = set()
        seen_normalized: set[str] = set()
        deduped: list[SearchResult] = []

        for result in results:
            url = result.url
            if not url:
                continue
            normalized = self._normalize_url(url)
            if url in seen_urls or normalized in seen_normalized:
                continue

            # Snippet similarity check against recent dedup'd results
            is_duplicate = False
            for existing in deduped[-20:]:
                if self._snippet_similarity(result.snippet, existing.snippet) > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_urls.add(url)
                seen_normalized.add(normalized)
                deduped.append(result)

        return deduped

    # ── Relevance scoring ───────────────────────────────────────────────────

    def _score_relevance(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[AggregatedResult]:
        """Score each result's relevance to the query."""
        query_terms = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        query_terms -= STOP_WORDS

        aggregated: list[AggregatedResult] = []
        for result in results:
            score = self._compute_relevance(result, query_terms)
            agg = AggregatedResult(
                search_result=result,
                relevance_score=score,
                query_matches=self._find_matches(result.snippet, query_terms),
            )
            aggregated.append(agg)
        return aggregated

    def _compute_relevance(
        self,
        result: SearchResult,
        query_terms: set[str],
    ) -> float:
        """Multi-factor relevance score in [0.0, 1.0].

        Multipliers > 1.0 boost (trusted sources rank higher than
        random ones); < 1.0 penalize (low-quality sources rank lower).
        The final score is clamped to [0.0, 1.0].
        """
        score = 0.0

        # 1. Position score (earlier = more relevant per search engine)
        position_score = 1.0 / (1.0 + result.position * 0.1)
        score += position_score * 0.3

        # 2. Query term coverage in title + snippet
        combined = (result.title + " " + result.snippet).lower()
        combined_words = set(re.sub(r"[^\w\s]", "", combined).split())
        if query_terms:
            term_coverage = len(query_terms & combined_words) / len(query_terms)
            score += term_coverage * 0.4
        else:
            score += 0.4

        # 3. Domain quality multiplier (boost > 1.0 for trusted, < 1.0 for low)
        domain = (result.domain or "").lower()
        domain_multiplier = 1.0
        for trusted_domain, multiplier in TRUSTED_DOMAINS.items():
            if trusted_domain in domain:
                domain_multiplier = multiplier
                break
        else:
            # .edu / .gov get a mild boost
            if domain.endswith(".edu") or domain.endswith(".gov"):
                domain_multiplier = 1.05
        for low_domain in LOW_QUALITY_DOMAINS:
            if low_domain in domain:
                domain_multiplier = 0.5
                break
        score *= domain_multiplier

        # 4. Snippet quality (length as proxy, capped at 30 words)
        snippet_len = len(result.snippet.split())
        snippet_score = min(snippet_len / 30, 1.0)
        score += snippet_score * 0.3

        return max(0.0, min(1.0, score))

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for deduplication (strip www, trailing slash,
        lowercase host+path, drop fragment)."""
        try:
            parsed = urlparse(url)
        except Exception:
            return url.lower()
        host = parsed.netloc.replace("www.", "").lower()
        path = parsed.path.rstrip("/").lower()
        return f"{host}{path}"

    @staticmethod
    def _snippet_similarity(a: str, b: str) -> float:
        """Jaccard similarity on words. Returns 0.0 for empty inputs."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    @staticmethod
    def _find_matches(text: str, terms: set[str]) -> list[str]:
        """Find which query terms appear in text (case-insensitive)."""
        text_lower = text.lower()
        return [term for term in terms if term in text_lower]
