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
    "AggregatedResult",
    # Provider base + impls
    "BaseSearchProvider",
    "BingSearchProvider",
    "BraveSearchProvider",
    "ContentExtractor",
    "DuckDuckGoProvider",
    "ExtractedContent",
    "FallbackSearchEngine",
    # Aggregation
    "SearchAggregator",
    "SearchResponse",
    "SearchResult",
    "SerpAPIProvider",
    # Extraction
    "extractor",
    # Engine + types
    "search_engine",
]
