"""
Multi-provider web search engine.

Supports:
  - DuckDuckGo (free, no API key)
  - SerpAPI    (Google results, paid)
  - Brave Search (privacy-focused, paid)
  - Bing Search  (Microsoft, paid)

All providers return a unified SearchResult schema.
Falls back to next provider if primary fails.
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus

import aiohttp
from loguru import logger

from config.settings import settings
from src.utils.cache import cache, make_cache_key


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

    def __post_init__(self):
        if not self.domain and self.url:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            self.domain = parsed.netloc.replace("www.", "")

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "position": self.position,
            "provider": self.provider,
            "domain": self.domain,
            "published_date": self.published_date,
        }


@dataclass
class SearchResponse:
    """Response from a search query."""
    query: str
    results: list[SearchResult]
    provider: str
    total_results: int = 0
    search_time_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None

    @property
    def has_results(self) -> bool:
        return len(self.results) > 0


# ── Abstract provider ─────────────────────────────────────────────────────────

class BaseSearchProvider(ABC):
    """Abstract base for all search providers."""

    def __init__(self, max_results: int = 10, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout

    @abstractmethod
    async def search(self, query: str) -> SearchResponse:
        """Execute search and return unified results."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider has valid credentials."""
        ...


# ── DuckDuckGo Provider ───────────────────────────────────────────────────────

class DuckDuckGoProvider(BaseSearchProvider):
    """
    DuckDuckGo search via their unofficial API.
    Free, no API key required. Rate limits apply.
    """

    BASE_URL = "https://html.duckduckgo.com/html/"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Import here to avoid hard dependency
        try:
            from duckduckgo_search import AsyncDDGS
            self._ddgs_class = AsyncDDGS
            self._available = True
        except ImportError:
            logger.warning("duckduckgo_search not installed")
            self._available = False

    @property
    def name(self) -> str:
        return "duckduckgo"

    @property
    def is_available(self) -> bool:
        return self._available

    async def search(self, query: str) -> SearchResponse:
        """Search DuckDuckGo using the duckduckgo-search library."""
        start = time.time()
        results: list[SearchResult] = []

        try:
            async with self._ddgs_class(
                timeout=self.timeout
            ) as ddgs:
                raw_results = await ddgs.atext(
                    query,
                    max_results=self.max_results,
                    safesearch="moderate",
                )

            for i, r in enumerate(raw_results or []):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    position=i + 1,
                    provider=self.name,
                    published_date=r.get("published", None),
                ))

        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=len(results),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── SerpAPI Provider ──────────────────────────────────────────────────────────

class SerpAPIProvider(BaseSearchProvider):
    """
    Google Search results via SerpAPI.
    Requires SERPAPI_KEY. Most reliable Google results.
    """

    BASE_URL = "https://serpapi.com/search.json"

    @property
    def name(self) -> str:
        return "serpapi"

    @property
    def is_available(self) -> bool:
        return bool(settings.search.serpapi_key)

    async def search(self, query: str) -> SearchResponse:
        start = time.time()
        results: list[SearchResult] = []

        params = {
            "q": query,
            "api_key": settings.search.serpapi_key,
            "num": self.max_results,
            "hl": "en",
            "gl": "us",
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(
                    self.BASE_URL, params=params
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            organic = data.get("organic_results", [])
            for i, r in enumerate(organic[:self.max_results]):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    position=i + 1,
                    provider=self.name,
                    published_date=r.get("date", None),
                ))

            # Also grab knowledge graph if available
            kg = data.get("knowledge_graph", {})
            if kg and kg.get("description"):
                results.insert(0, SearchResult(
                    title=kg.get("title", "Knowledge Graph"),
                    url=kg.get("source", {}).get("link", ""),
                    snippet=kg.get("description", ""),
                    position=0,
                    provider=f"{self.name}_kg",
                ))

        except Exception as e:
            logger.warning(f"SerpAPI search error: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=data.get("search_information", {}).get(
                "total_results", len(results)
            ),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── Brave Search Provider ─────────────────────────────────────────────────────

class BraveSearchProvider(BaseSearchProvider):
    """
    Brave Search API - privacy-focused, independent index.
    Requires BRAVE_SEARCH_API_KEY.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    @property
    def name(self) -> str:
        return "brave"

    @property
    def is_available(self) -> bool:
        return bool(settings.search.brave_search_api_key)

    async def search(self, query: str) -> SearchResponse:
        start = time.time()
        results: list[SearchResult] = []

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

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers,
            ) as session:
                async with session.get(
                    self.BASE_URL, params=params
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            web_results = (
                data.get("web", {}).get("results", [])
            )
            for i, r in enumerate(web_results[:self.max_results]):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description", ""),
                    position=i + 1,
                    provider=self.name,
                    published_date=r.get("age", None),
                ))

        except Exception as e:
            logger.warning(f"Brave search error: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=len(results),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── Bing Search Provider ──────────────────────────────────────────────────────

class BingSearchProvider(BaseSearchProvider):
    """
    Microsoft Bing Search API.
    Requires BING_SEARCH_API_KEY.
    """

    BASE_URL = "https://api.bing.microsoft.com/v7.0/search"

    @property
    def name(self) -> str:
        return "bing"

    @property
    def is_available(self) -> bool:
        return bool(settings.search.bing_search_api_key)

    async def search(self, query: str) -> SearchResponse:
        start = time.time()
        results: list[SearchResult] = []

        headers = {
            "Ocp-Apim-Subscription-Key": settings.search.bing_search_api_key
        }
        params = {
            "q": query,
            "count": self.max_results,
            "mkt": "en-US",
            "responseFilter": "Webpages",
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=headers,
            ) as session:
                async with session.get(
                    self.BASE_URL, params=params
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            web_pages = data.get("webPages", {}).get("value", [])
            for i, r in enumerate(web_pages[:self.max_results]):
                results.append(SearchResult(
                    title=r.get("name", ""),
                    url=r.get("url", ""),
                    snippet=r.get("snippet", ""),
                    position=i + 1,
                    provider=self.name,
                    published_date=r.get("dateLastCrawled", None),
                ))

        except Exception as e:
            logger.warning(f"Bing search error: {e}")
            return SearchResponse(
                query=query,
                results=[],
                provider=self.name,
                error=str(e),
                search_time_ms=(time.time() - start) * 1000,
            )

        return SearchResponse(
            query=query,
            results=results,
            provider=self.name,
            total_results=data.get("webPages", {}).get(
                "totalEstimatedMatches", len(results)
            ),
            search_time_ms=(time.time() - start) * 1000,
        )


# ── Fallback Search Engine ────────────────────────────────────────────────────

class FallbackSearchEngine:
    """
    Multi-provider search with automatic fallback.

    Priority order:
      1. Configured primary provider
      2. Any available provider
      3. DuckDuckGo as last resort

    Results are cached to reduce redundant API calls.
    """

    def __init__(self):
        self._providers: list[BaseSearchProvider] = [
            DuckDuckGoProvider(
                max_results=settings.search.max_search_results
            ),
            SerpAPIProvider(
                max_results=settings.search.max_search_results
            ),
            BraveSearchProvider(
                max_results=settings.search.max_search_results
            ),
            BingSearchProvider(
                max_results=settings.search.max_search_results
            ),
        ]
        # Build ordered list: configured provider first
        self._ordered = self._build_provider_order()

    def _build_provider_order(self) -> list[BaseSearchProvider]:
        """Order providers: configured primary first, then available ones."""
        primary_name = settings.search.search_provider
        by_name = {p.name: p for p in self._providers}

        ordered = []
        if primary_name in by_name:
            ordered.append(by_name[primary_name])

        for p in self._providers:
            if p not in ordered and p.is_available:
                ordered.append(p)

        # Always have DuckDuckGo as final fallback
        ddg = by_name.get("duckduckgo")
        if ddg and ddg not in ordered:
            ordered.append(ddg)

        logger.info(
            f"Search provider order: "
            f"{[p.name for p in ordered]}"
        )
        return ordered

    async def search(
        self,
        query: str,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ) -> SearchResponse:
        """
        Search with fallback. Returns first successful response.
        Caches results to avoid duplicate API calls.
        """
        if use_cache:
            cache_key = make_cache_key("search", query)
            cached = await cache.get(cache_key)
            if cached:
                logger.debug(f"Search cache hit: {query[:50]}")
                # Reconstruct SearchResponse from dict
                response = SearchResponse(
                    query=cached["query"],
                    results=[SearchResult(**r) for r in cached["results"]],
                    provider=cached["provider"],
                    total_results=cached["total_results"],
                    cached=True,
                )
                return response

        for provider in self._ordered:
            try:
                logger.info(
                    f"Searching [{provider.name}]: {query[:80]}"
                )
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

                logger.warning(
                    f"Provider {provider.name} returned no results, "
                    f"trying next..."
                )

            except Exception as e:
                logger.warning(
                    f"Provider {provider.name} failed: {e}, "
                    f"trying next..."
                )
                continue

        # All providers failed
        logger.error(f"All search providers failed for: {query}")
        return SearchResponse(
            query=query,
            results=[],
            provider="none",
            error="All search providers failed",
        )

    async def multi_search(
        self,
        queries: list[str],
        use_cache: bool = True,
    ) -> list[SearchResponse]:
        """Search multiple queries concurrently."""
        tasks = [
            self.search(q, use_cache=use_cache)
            for q in queries
        ]
        return await asyncio.gather(*tasks)


# Singleton
search_engine = FallbackSearchEngine()