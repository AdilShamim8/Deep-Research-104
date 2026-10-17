"""
Web search, content extraction, and result aggregation.

Public surface:
    - search_engine, FallbackSearchEngine, SearchResult, SearchResponse
    - extractor, ExtractedContent, ContentExtractor
    - SearchAggregator, AggregatedResult

Date tag: 17 October 2026.
"""

from src.search.content_extractor import (
    ContentExtractor,
    ExtractedContent,
    extractor,
)
from src.search.search_aggregator import (
    AggregatedResult,
    SearchAggregator,
)
from src.search.web_search import (
    BaseSearchProvider,
    BingSearchProvider,
    BraveSearchProvider,
    DuckDuckGoProvider,
    FallbackSearchEngine,
    SearchResponse,
    SearchResult,
    SerpAPIProvider,
    search_engine,
)

__all__ = [
    # Engine + types
    "search_engine",
    "FallbackSearchEngine",
    "SearchResult",
    "SearchResponse",
    # Provider base + impls
    "BaseSearchProvider",
    "DuckDuckGoProvider",
    "SerpAPIProvider",
    "BraveSearchProvider",
    "BingSearchProvider",
    # Extraction
    "extractor",
    "ContentExtractor",
    "ExtractedContent",
    # Aggregation
    "SearchAggregator",
    "AggregatedResult",
]
