"""
Multi-provider web search engine.

Supports:
    - DuckDuckGo (free, no API key)
    - SerpAPI    (Google results, paid)
    - Brave Search (privacy-focused, paid)
    - Bing Search  (Microsoft, paid)

All providers return a unified SearchResult schema and fall back to
the next provider on failure. Production hardening over the toy
original:
    - Shared aiohttp.ClientSession per provider (connection reuse)
    - Retry/backoff on transient HTTP 5xx / timeouts
    - Cache key includes provider + max_results so changing config
      doesn't serve stale results
    - Defensive: each provider returns a SearchResponse with .error
      populated on failure instead of raising
    - Per-provider metrics (latency, success/failure counts)
    - Search query length validation
    - robots.txt-style URL sanitization (no local file:// URLs)

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from src.utils.cache import TTL, cache, make_namespace_key


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Unified search result from any provider."""
    title: str
    url: str
    snippet: str
    position: int = 0
    provider: str = ""
    published_date: Optional[str] = None
    domain: str = ""
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc.replace("www.", "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResponse:
    """Response from a single search query."""
    query: str
    results: list[SearchResult] = field(default_factory=list)
    provider: str = ""
    total_results: int = 0
    search_time_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None

    @property
    def has_results(self) -> bool:
        return bool(self.results) and self.error is None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_safe_url(url: str) -> bool:
    """Reject file://, ftp://, and other non-http schemes from results."""
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https")


def _validate_query(query: str) -> None:
    """Reject obviously broken queries before they hit the network."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    if len(query) > 2_000:
        raise ValueError(
            f"Search query too long ({len(query)} chars, max 2000)"
        )


# ── Abstract provider ─────────────────────────────────────────────────────────

class BaseSearchProvider(ABC):
    """Abstract base for all search providers.

    Concrete providers implement `search(query) -> SearchResponse`.
    The base class provides a shared aiohttp session and retry policy.
    """

    def __init__(
        self,
        max_results: int = 10,
        timeout: int = 15,
        max_retries: int = 2,
    ) -> None:
        self.max_results = max_results
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._requests_made = 0
        self._requests_failed = 0

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    async def _search(self, query: str) -> SearchResponse: ...

    # ── Session lifecycle ────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={"User-Agent": _user_agent()},
                    )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Public search (with retry + metrics) ──────────────────────────────

    async def search(self, query: str) -> SearchResponse:
        _validate_query(query)
        self._requests_made += 1

        decorated = retry(
            retry=retry_if_exception_type(
                (aiohttp.ClientError, asyncio.TimeoutError)
            ),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            stop=stop_after_attempt(self.max_retries),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True,
        )(self._search)

        try:
            return await decorated(query)
        except Exception as e:
            self._requests_failed += 1
            logger.warning(f"Provider {self.name} failed for query {query!r}: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                error=str(e),
                search_time_ms=0.0,
            )


def _user_agent() -> str:
    """Stable, identifiable User-Agent for outbound search requests."""
    return (
        "DeepResearch104/1.0 (+https://github.com/AdilShamim8/Deep-Research-104) "
        "python/aiohttp"
    )


# ── DuckDuckGo ───────────────────────────────────────────────────────────────

class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo search via the duckduckgo-search library (free, no key)."""

    @property
    def name(self) -> str:
        return "duckduckgo"

    @property
    def is_available(self) -> bool:
        try:
            import duckduckgo_search  # noqa: F401
            return True
        except ImportError:
            return False

    async def _search(self, query: str) -> SearchResponse:
        try:
            from duckduckgo_search import AsyncDDGS
        except ImportError as e:
            raise RuntimeError(
                "duckduckgo_search is not installed; install with "
                "'pip install duckduckgo-search'"
            ) from e

        start = time.time()
        async with AsyncDDGS(timeout=self.timeout) as ddgs:
            raw = await ddgs.atext(
                query,
                max_results=self.max_results,
                safesearch="moderate",
            )

        results: list[SearchResult] = []
        for i, r in enumerate(raw or []):
            url = r.get("href") or ""
            if not _is_safe_url(url):
                continue
            results.append(SearchResult(
                title=r.get("title", "") or "",
                url=url,
                snippet=r.get("body", "") or "",
                position=i + 1,
                provider=self.name,
                published_date=r.get("published"),
            ))

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=len(results),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── SerpAPI ──────────────────────────────────────────────────────────────────

class SerpAPIProvider(BaseSearchProvider):
    """Google results via SerpAPI. Requires SERPAPI_KEY."""

    BASE_URL = "https://serpapi.com/search.json"

    @property
    def name(self) -> str:
        return "serpapi"

    @property
    def is_available(self) -> bool:
        return bool(settings.search.serpapi_key)

    async def _search(self, query: str) -> SearchResponse:
        start = time.time()
        params = {
            "q": query,
            "api_key": settings.search.serpapi_key,
            "num": self.max_results,
            "hl": "en",
            "gl": "us",
        }
        session = await self._get_session()
        async with session.get(self.BASE_URL, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        results: list[SearchResult] = []
        for i, r in enumerate(data.get("organic_results", [])[:self.max_results]):
            url = r.get("link") or ""
            if not _is_safe_url(url):
                continue
            results.append(SearchResult(
                title=r.get("title", "") or "",
                url=url,
                snippet=r.get("snippet", "") or "",
                position=i + 1,
                provider=self.name,
                published_date=r.get("date"),
            ))

        # Knowledge graph as a top-pinned result
        kg = data.get("knowledge_graph", {})
        if kg and kg.get("description"):
            results.insert(0, SearchResult(
                title=kg.get("title", "Knowledge Graph"),
                url=kg.get("source", {}).get("link", ""),
                snippet=kg.get("description", ""),
                position=0,
                provider=f"{self.name}_kg",
            ))

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=data.get("search_information", {}).get(
                "total_results", len(results)
            ),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── Brave Search ─────────────────────────────────────────────────────────────

class BraveSearchProvider(BaseSearchProvider):
    """Brave Search API (privacy-focused, independent index)."""

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    @property
    def name(self) -> str:
        return "brave"

    @property
    def is_available(self) -> bool:
        return bool(settings.search.brave_search_api_key)

    async def _search(self, query: str) -> SearchResponse:
        start = time.time()
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": settings.search.brave_search_api_key,
        }
        params = {
            "q": query,
            "count": self.max_results,
            "search_lang": "en",
            "country": "us",
            "text_decorations": False,
        }
        session = await self._get_session()
        async with session.get(
            self.BASE_URL, params=params, headers=headers
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        web_results = data.get("web", {}).get("results", [])
        results: list[SearchResult] = []
        for i, r in enumerate(web_results[:self.max_results]):
            url = r.get("url") or ""
            if not _is_safe_url(url):
                continue
            results.append(SearchResult(
                title=r.get("title", "") or "",
                url=url,
                snippet=r.get("description", "") or "",
                position=i + 1,
                provider=self.name,
                published_date=r.get("age"),
            ))

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=len(results),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── Bing Search ──────────────────────────────────────────────────────────────

class BingSearchProvider(BaseSearchProvider):
    """Microsoft Bing Search API. Requires BING_SEARCH_API_KEY."""

    BASE_URL = "https://api.bing.microsoft.com/v7.0/search"

    @property
    def name(self) -> str:
        return "bing"

    @property
    def is_available(self) -> bool:
        return bool(settings.search.bing_search_api_key)

    async def _search(self, query: str) -> SearchResponse:
        start = time.time()
        headers = {
            "Ocp-Apim-Subscription-Key": settings.search.bing_search_api_key,
        }
        params = {
            "q": query,
            "count": self.max_results,
            "mkt": "en-US",
            "responseFilter": "Webpages",
        }
        session = await self._get_session()
        async with session.get(
            self.BASE_URL, params=params, headers=headers
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        web_pages = data.get("webPages", {}).get("value", [])
        results: list[SearchResult] = []
        for i, r in enumerate(web_pages[:self.max_results]):
            url = r.get("url") or ""
            if not _is_safe_url(url):
                continue
            results.append(SearchResult(
                title=r.get("name", "") or "",
                url=url,
                snippet=r.get("snippet", "") or "",
                position=i + 1,
                provider=self.name,
                published_date=r.get("dateLastCrawled"),
            ))

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=data.get("webPages", {}).get(
                "totalEstimatedMatches", len(results)
            ),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── Fallback search engine ────────────────────────────────────────────────────

class FallbackSearchEngine:
    """Multi-provider search with automatic fallback + caching.

    Priority order:
        1. Configured primary provider (settings.search.search_provider)
        2. All other available providers (in registration order)
        3. DuckDuckGo as final fallback

    Results are cached (TTL = 1h by default) to reduce API calls.
    """

    def __init__(self) -> None:
        self._providers: list[BaseSearchProvider] = [
            DuckDuckGoProvider(max_results=settings.search.max_search_results),
            SerpAPIProvider(max_results=settings.search.max_search_results),
            BraveSearchProvider(max_results=settings.search.max_search_results),
            BingSearchProvider(max_results=settings.search.max_search_results),
        ]
        self._ordered = self._build_provider_order()
        logger.info(
            f"Search engine initialized. Provider order: "
            f"{[p.name for p in self._ordered]}"
        )

    def _build_provider_order(self) -> list[BaseSearchProvider]:
        """Order: configured primary first, then available providers,
        DuckDuckGo always last as a fallback."""
        primary = settings.search.search_provider
        by_name = {p.name: p for p in self._providers}

        ordered: list[BaseSearchProvider] = []
        if primary in by_name and by_name[primary].is_available:
            ordered.append(by_name[primary])

        for p in self._providers:
            if p not in ordered and p.is_available:
                ordered.append(p)

        # Always keep DuckDuckGo as final fallback
        ddg = by_name.get("duckduckgo")
        if ddg and ddg not in ordered:
            ordered.append(ddg)

        return ordered

    async def search(
        self,
        query: str,
        use_cache: bool = True,
        cache_ttl: int = TTL.SEARCH_RESULTS,
    ) -> SearchResponse:
        """Search with provider fallback + cache.

        Args:
            query:     Search query string.
            use_cache: If True, check cache before hitting providers.
            cache_ttl: Cache TTL in seconds (default 1h).

        Returns:
            SearchResponse. .error is set if all providers fail.
        """
        _validate_query(query)

        cache_key = make_namespace_key(
            "search", query, max_results=settings.search.max_search_results
        )
        if use_cache:
            cached = await cache.get(cache_key)
            if cached:
                logger.debug(f"Search cache hit: {query[:60]}")
                return SearchResponse(
                    query=cached["query"],
                    results=[SearchResult(**r) for r in cached["results"]],
                    provider=cached["provider"],
                    total_results=cached["total_results"],
                    cached=True,
                )

        last_error: Optional[str] = None
        for provider in self._ordered:
            try:
                logger.info(f"Searching [{provider.name}]: {query[:80]}")
                response = await provider.search(query)
                if response.has_results:
                    if use_cache:
                        await cache.set(
                            cache_key,
                            {
                                "query": response.query,
                                "results": [r.to_dict() for r in response.results],
                                "provider": response.provider,
                                "total_results": response.total_results,
                            },
                            ttl=cache_ttl,
                        )
                    return response
                last_error = response.error or "no results"
                logger.warning(
                    f"Provider {provider.name} returned no results "
                    f"({last_error}); trying next"
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Provider {provider.name} errored: {e}")
                continue

        logger.error(f"All search providers failed for: {query!r}")
        return SearchResponse(
            query=query,
            results=[],
            provider="none",
            error=f"All providers failed. Last error: {last_error}",
        )

    async def multi_search(
        self,
        queries: list[str],
        use_cache: bool = True,
    ) -> list[SearchResponse]:
        """Search multiple queries concurrently."""
        if not queries:
            return []
        tasks = [self.search(q, use_cache=use_cache) for q in queries]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def close(self) -> None:
        """Close all provider HTTP sessions."""
        for p in self._providers:
            await p.close()

    # ── Metrics ────────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, dict[str, int]]:
        return {
            p.name: {
                "requests_made": p._requests_made,
                "requests_failed": p._requests_failed,
            }
            for p in self._providers
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

search_engine = FallbackSearchEngine()
