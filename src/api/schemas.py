"""
Pydantic v2 schemas for the REST API.

Request/response validation, serialization, and OpenAPI documentation.
Production hardening over the toy original:
    - Strict config (extra='forbid' on requests to catch typos early)
    - All string fields have explicit max_length (prevents DoS via
      oversized payloads)
    - field_validator for query/question normalization (trim whitespace)
    - Comprehensive docstrings + examples surfaced in OpenAPI docs
    - ErrorResponse includes request_id, error_code, timestamp
    - Health/Version response schemas added

Date tag: 17 October 2026.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Provider + mode aliases ────────────────────────────────────────────────────

ProviderName = Literal["openai", "deepseek", "ollama", "vllm", "llamacpp"]
SearchProviderName = Literal["serpapi", "brave", "bing", "duckduckgo"]
Confidence = Literal["low", "medium", "high"]
ResearchMode = Literal[
    "deep_research",
    "meta_cot",
    "cot",
    "tot",
    "self_consistency",
    "parallel",
]


# ── Request schemas ───────────────────────────────────────────────────────────


class _StrictRequest(BaseModel):
    """Base for all request schemas: forbid unknown fields (fail on typos)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearchRequest(_StrictRequest):
    """Deep research request — the flagship endpoint."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=2_000,
        description="Research question to investigate.",
        examples=["What are the latest breakthroughs in fusion energy?"],
    )
    provider: ProviderName = Field(
        default="openai",
        description="LLM provider to use for synthesis.",
    )
    model_name: str | None = Field(
        default=None,
        max_length=128,
        description="Specific model override (e.g. 'o3-mini', 'deepseek-reasoner').",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum research iterations (search -> reason -> refine).",
    )
    fast_mode: bool = Field(
        default=False,
        description="Fast mode: single iteration, fewer queries, lower latency.",
    )
    mode: ResearchMode = Field(
        default="deep_research",
        description="Research pipeline to run.",
    )

    @field_validator("question")
    @classmethod
    def _normalize_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty/whitespace")
        return v


class CoTRequest(_StrictRequest):
    """Chain-of-Thought reasoning request."""

    question: str = Field(
        ..., min_length=3, max_length=2_000, description="Question to reason about."
    )
    context: str | None = Field(
        default=None,
        max_length=20_000,
        description="Optional background context (e.g. from web search).",
    )
    cot_mode: Literal["zero_shot", "few_shot", "structured"] = Field(
        default="structured",
        description="CoT prompting strategy.",
    )
    n_samples: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of independent reasoning paths (self-consistency).",
    )
    provider: ProviderName = Field(default="openai")
    model_name: str | None = Field(default=None, max_length=128)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("question")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return v.strip()


class ToTRequest(_StrictRequest):
    """Tree of Thoughts search request."""

    problem: str = Field(
        ..., min_length=5, max_length=2_000, description="Problem to solve via ToT search."
    )
    context: str | None = Field(default=None, max_length=10_000)
    branching_factor: int = Field(default=3, ge=2, le=6)
    max_depth: int = Field(default=4, ge=1, le=8)
    beam_width: int = Field(default=2, ge=1, le=5)
    strategy: Literal["beam", "bfs", "dfs"] = Field(default="beam")
    provider: ProviderName = Field(default="openai")
    model_name: str | None = Field(default=None, max_length=128)

    @field_validator("problem")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return v.strip()


class SearchRequest(_StrictRequest):
    """Web search request."""

    query: str = Field(..., min_length=2, max_length=500, description="Search query.")
    max_results: int = Field(default=10, ge=1, le=50)
    fetch_content: bool = Field(
        default=False,
        description="If True, fetch full page content for top results.",
    )
    provider: SearchProviderName | None = Field(
        default=None,
        description="Force a specific search provider (default: auto-fallback).",
    )

    @field_validator("query")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return v.strip()


# ── Response schemas ──────────────────────────────────────────────────────────


class SourceSchema(BaseModel):
    """A single research source cited in a report."""

    model_config = ConfigDict(from_attributes=True)
    title: str
    url: str
    domain: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    snippet: str = ""
    published_date: str | None = None


class ResearchResponse(BaseModel):
    """Deep research response."""

    question: str
    answer: str
    confidence: Confidence
    sources: list[SourceSchema]
    iterations: int
    total_sources_found: int
    total_tokens_used: int
    synthesis_model: str
    processing_time_ms: float
    request_id: str | None = None

    @classmethod
    def from_report(
        cls,
        report: Any,
        processing_time_ms: float = 0.0,
        request_id: str | None = None,
    ) -> ResearchResponse:
        """Build a ResearchResponse from a ResearchReport dataclass."""
        return cls(
            question=report.question,
            answer=report.answer,
            confidence=report.confidence,
            sources=[
                SourceSchema(
                    title=s.title,
                    url=s.url,
                    domain=s.search_result.domain,
                    relevance_score=s.relevance_score,
                    snippet=s.search_result.snippet,
                    published_date=s.search_result.published_date,
                )
                for s in report.sources[:10]
            ],
            iterations=len(report.iterations),
            total_sources_found=report.total_sources_found,
            total_tokens_used=report.total_tokens_used,
            synthesis_model=report.synthesis_model,
            processing_time_ms=processing_time_ms,
            request_id=request_id,
        )


class CoTResponse(BaseModel):
    """Chain-of-Thought response."""

    answer: str
    reasoning: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    supporting_votes: int = 1
    total_votes: int = 1
    model: str = ""
    request_id: str | None = None


class ToTResponse(BaseModel):
    """Tree of Thoughts response."""

    answer: str
    best_score: float = Field(ge=0.0, le=1.0)
    total_nodes_explored: int
    search_strategy: str
    path_length: int
    model: str = ""
    request_id: str | None = None


class SearchResultSchema(BaseModel):
    """A single search result."""

    model_config = ConfigDict(from_attributes=True)
    title: str
    url: str
    snippet: str
    domain: str
    position: int
    provider: str
    published_date: str | None = None


class SearchResponse(BaseModel):
    """Web search response."""

    query: str
    results: list[SearchResultSchema]
    total_results: int
    provider: str
    search_time_ms: float
    cached: bool = False
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str = Field(description="Short error code/category.")
    detail: str = Field(default="", description="Human-readable detail.")
    request_id: str = Field(default="", description="Correlation ID.")
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code (e.g. 'rate_limited').",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp of the error.",
    )

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        request_id: str = "",
        error_code: str | None = None,
        detail: str | None = None,
    ) -> ErrorResponse:
        return cls(
            error=type(exc).__name__,
            detail=detail or str(exc),
            request_id=request_id,
            error_code=error_code,
            timestamp=datetime.now(UTC).isoformat(),
        )


# ── Health + version ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health-check response."""

    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    python_version: str
    version: str
    build_date: str
    redis: Literal["connected", "disconnected"] = "disconnected"
    uptime_seconds: float = 0.0
    timestamp: float


class VersionResponse(BaseModel):
    """Version info response."""

    name: str
    version: str
    build_date: str
    python_version: str
    app_env: str


class ReadyResponse(BaseModel):
    """Readiness probe response — all subsystems must report ready."""

    status: Literal["ready", "not_ready"] = "ready"
    checks: dict[str, bool] = Field(default_factory=dict)
    timestamp: float
