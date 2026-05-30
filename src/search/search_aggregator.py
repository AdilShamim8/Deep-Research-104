"""
Search result aggregation, deduplication, and relevance ranking.

Takes raw search results from multiple queries/providers,
cleans them up, removes duplicates, and ranks by relevance.
"""

import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse
from loguru import logger

from src.search.web_search import SearchResult, SearchResponse
from src.search.content_extractor import ContentExtractor, ExtractedContent
from src.utils.token_counter import TokenCounter


@dataclass
class AggregatedResult:
    """
    A deduplicated, ranked search result with extracted content.
    """
    search_result: SearchResult
    content: Optional[ExtractedContent] = None
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
        return (
            self.content.title
            if self.has_content and self.content.title
            else self.search_result.title
        )

    @property
    def text(self) -> str:
        if self.has_content:
            return self.content.text
        return self.search_result.snippet

    def to_context_string(self, max_length: int = 3000) -> str:
        """Format result for LLM context."""
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


class SearchAggregator:
    """
    Aggregates and ranks search results from multiple queries.

    Pipeline:
      1. Collect results from multiple search responses
      2. Deduplicate by URL and content similarity
      3. Score relevance against the original query
      4. Fetch full content for top results
      5. Return ranked, enriched results
    """

    def __init__(
        self,
        extractor: Optional[ContentExtractor] = None,
        max_results_to_fetch: int = 5,
        token_counter: Optional[TokenCounter] = None,
    ):
        self._extractor = extractor or ContentExtractor()
        self.max_results_to_fetch = max_results_to_fetch
        self._token_counter = token_counter or TokenCounter()

        # Trusted domains get a relevance boost
        self._trusted_domains = {
            "wikipedia.org": 0.9,
            "arxiv.org": 0.95,
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "nature.com": 0.9,
            "science.org": 0.9,
            "github.com": 0.85,
            "stackoverflow.com": 0.8,
            "docs.python.org": 0.85,
            "nytimes.com": 0.75,
            "bbc.com": 0.75,
            "reuters.com": 0.8,
            "apnews.com": 0.8,
        }

        # Low-quality domains get penalized
        self._low_quality_domains = {
            "pinterest.com", "quora.com",
            "answers.com", "ask.com",
        }

    # ── Public API ────────────────────────────────────────────────────────────

    async def aggregate(
        self,
        query: str,
        search_responses: list[SearchResponse],
        fetch_content: bool = True,
        max_total_results: int = 10,
    ) -> list[AggregatedResult]:
        """
        Aggregate multiple search responses into ranked results.

        Args:
            query:              Original research query.
            search_responses:   List of SearchResponse objects.
            fetch_content:      Whether to fetch full page content.
            max_total_results:  Maximum results to return.

        Returns:
            Ranked list of AggregatedResult.
        """
        # Collect all results
        all_results = []
        for response in search_responses:
            all_results.extend(response.results)

        logger.info(
            f"Aggregating {len(all_results)} results "
            f"from {len(search_responses)} searches"
        )

        # Deduplicate
        deduped = self._deduplicate(all_results)
        logger.debug(
            f"After dedup: {len(deduped)} results "
            f"(removed {len(all_results) - len(deduped)})"
        )

        # Score relevance
        scored = self._score_relevance(query, deduped)

        # Sort by relevance
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        scored = scored[:max_total_results]

        # Fetch content for top results
        if fetch_content:
            top_n = min(self.max_results_to_fetch, len(scored))
            urls = [r.url for r in scored[:top_n]]
            contents = await self._extractor.extract_many(urls)

            for agg_result, content in zip(scored[:top_n], contents):
                if content.success:
                    agg_result.content = content

        successful_content = sum(1 for r in scored if r.has_content)
        logger.info(
            f"Aggregation complete: {len(scored)} results, "
            f"{successful_content} with full content"
        )

        return scored

    def build_context_string(
        self,
        results: list[AggregatedResult],
        max_total_tokens: int = 12000,
        model: str = "gpt-4o",
    ) -> str:
        """
        Build a single context string from aggregated results
        that fits within the token budget.

        Distributes token budget across results proportionally
        to their relevance score.
        """
        if not results:
            return "No search results found."

        counter = TokenCounter(model)
        header = "## Search Results\n\n"
        header_tokens = counter.count_tokens(header)
        budget = max_total_tokens - header_tokens

        # Proportional token budget
        total_score = sum(r.relevance_score for r in results) or 1.0
        sections = []

        remaining_budget = budget
        for result in results:
            if remaining_budget <= 100:
                break

            # Allocate tokens proportional to score
            allocated = int(
                (result.relevance_score / total_score) * budget
            )
            allocated = min(allocated, remaining_budget, 4000)
            allocated = max(allocated, 200)  # Minimum per result

            section = result.to_context_string(max_length=allocated * 4)
            section_tokens = counter.count_tokens(section)

            if section_tokens > allocated:
                # Truncate
                section = counter.truncate_text_to_tokens(section, allocated)
                section_tokens = allocated

            sections.append(section)
            remaining_budget -= section_tokens

        context = header + "\n\n---\n\n".join(sections)

        final_tokens = counter.count_tokens(context)
        logger.debug(
            f"Context built: {len(sections)} sources, "
            f"{final_tokens}/{max_total_tokens} tokens"
        )

        return context

    # ── Private methods ───────────────────────────────────────────────────────

    def _deduplicate(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Remove duplicate results by:
        1. Exact URL match
        2. Normalized URL match (strip www, trailing slash, params)
        3. High snippet similarity
        """
        seen_urls: set[str] = set()
        seen_normalized: set[str] = set()
        deduped: list[SearchResult] = []

        for result in results:
            url = result.url
            normalized = self._normalize_url(url)

            if url in seen_urls or normalized in seen_normalized:
                continue

            # Check snippet similarity against recent results
            is_duplicate = False
            for existing in deduped[-20:]:  # Only check recent
                if self._snippet_similarity(
                    result.snippet, existing.snippet
                ) > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_urls.add(url)
                seen_normalized.add(normalized)
                deduped.append(result)

        return deduped

    def _score_relevance(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[AggregatedResult]:
        """Score each result's relevance to the query."""
        query_terms = set(
            re.sub(r'[^\w\s]', '', query.lower()).split()
        )
        # Remove stop words
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "is", "was", "are", "were",
            "what", "how", "why", "when", "where", "who",
        }
        query_terms -= stop_words

        aggregated = []
        for result in results:
            score = self._compute_relevance(result, query_terms)
            agg = AggregatedResult(
                search_result=result,
                relevance_score=score,
                query_matches=self._find_matches(
                    result.snippet, query_terms
                ),
            )
            aggregated.append(agg)

        return aggregated

    def _compute_relevance(
        self,
        result: SearchResult,
        query_terms: set[str],
    ) -> float:
        """Multi-factor relevance score."""
        score = 0.0

        # 1. Position score (earlier = more relevant per search engine)
        position_score = 1.0 / (1.0 + result.position * 0.1)
        score += position_score * 0.3

        # 2. Query term coverage in title + snippet
        combined = (result.title + " " + result.snippet).lower()
        combined_words = set(re.sub(r'[^\w\s]', '', combined).split())

        if query_terms:
            term_coverage = len(
                query_terms & combined_words
            ) / len(query_terms)
            score += term_coverage * 0.4
        else:
            score += 0.4  # No specific terms, neutral

        # 3. Domain quality boost/penalty
        domain = result.domain.lower()
        for trusted_domain, boost in self._trusted_domains.items():
            if trusted_domain in domain:
                score *= boost
                break
        for low_domain in self._low_quality_domains:
            if low_domain in domain:
                score *= 0.5
                break

        # 4. Snippet quality (length as proxy)
        snippet_len = len(result.snippet.split())
        snippet_score = min(snippet_len / 30, 1.0)
        score += snippet_score * 0.3

        return max(0.0, min(1.0, score))

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        host = parsed.netloc.replace("www.", "").lower()
        path = parsed.path.rstrip("/").lower()
        return f"{host}{path}"

    @staticmethod
    def _snippet_similarity(a: str, b: str) -> float:
        """Jaccard similarity on words."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _find_matches(text: str, terms: set[str]) -> list[str]:
        """Find which query terms appear in text."""
        text_lower = text.lower()
        return [term for term in terms if term in text_lower]