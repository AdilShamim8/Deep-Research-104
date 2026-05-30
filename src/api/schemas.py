"""
Pydantic schemas for the REST API.
Request/response validation and serialization.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    """Deep research request."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Research question",
        examples=["What are the latest breakthroughs in fusion energy?"]
    )
    provider: Literal["openai", "deepseek", "ollama"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model override"
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum research iterations"
    )
    fast_mode: bool = Field(
        default=False,
        description="Fast mode: single iteration, fewer queries"
    )
    mode: Literal[
        "deep_research",
        "meta_cot",
        "cot",
        "tot",
        "self_consistency",
        "parallel",
    ] = Field(
        default="deep_research",
        description="Research mode"
    )


class CoTRequest(BaseModel):
    """Chain-of-Thought request."""
    question: str = Field(..., min_length=3, max_length=2000)
    context: Optional[str] = Field(default=None, max_length=10000)
    cot_mode: Literal["zero_shot", "few_shot", "structured"] = Field(
        default="structured"
    )
    n_samples: int = Field(default=1, ge=1, le=20)
    provider: Literal["openai", "deepseek", "ollama"] = Field(
        default="openai"
    )
    model_name: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ToTRequest(BaseModel):
    """Tree of Thoughts request."""
    problem: str = Field(..., min_length=5, max_length=2000)
    context: Optional[str] = Field(default=None, max_length=5000)
    branching_factor: int = Field(default=3, ge=2, le=6)
    max_depth: int = Field(default=4, ge=1, le=8)
    beam_width: int = Field(default=2, ge=1, le=5)
    strategy: Literal["beam", "bfs", "dfs"] = Field(default="beam")
    provider: Literal["openai", "deepseek", "ollama"] = Field(
        default="openai"
    )
    model_name: Optional[str] = None


class SearchRequest(BaseModel):
    """Web search request."""
    query: str = Field(..., min_length=2, max_length=500)
    max_results: int = Field(default=10, ge=1, le=50)
    fetch_content: bool = Field(default=False)
    provider: Optional[Literal["serpapi", "brave", "bing", "duckduckgo"]] = None


# ── Response schemas ──────────────────────────────────────────────────────────

class SourceSchema(BaseModel):
    """A single research source."""
    title: str
    url: str
    domain: str
    relevance_score: float
    snippet: str = ""


class ResearchResponse(BaseModel):
    """Deep research response."""
    question: str
    answer: str
    confidence: Literal["low", "medium", "high"]
    sources: list[SourceSchema]
    iterations: int
    total_sources_found: int
    total_tokens_used: int
    synthesis_model: str
    processing_time_ms: float

    @classmethod
    def from_report(cls, report, processing_time_ms: float = 0.0):
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
                )
                for s in report.sources[:10]
            ],
            iterations=len(report.iterations),
            total_sources_found=report.total_sources_found,
            total_tokens_used=report.total_tokens_used,
            synthesis_model=report.synthesis_model,
            processing_time_ms=processing_time_ms,
        )


class CoTResponse(BaseModel):
    """Chain-of-Thought response."""
    answer: str
    reasoning: str
    confidence: float = 1.0
    supporting_votes: int = 1
    total_votes: int = 1
    model: str = ""


class ToTResponse(BaseModel):
    """Tree of Thoughts response."""
    answer: str
    best_score: float
    total_nodes_explored: int
    search_strategy: str
    path_length: int
    model: str = ""


class SearchResultSchema(BaseModel):
    """A single search result."""
    title: str
    url: str
    snippet: str
    domain: str
    position: int
    provider: str


class SearchResponse(BaseModel):
    """Web search response."""
    query: str
    results: list[SearchResultSchema]
    total_results: int
    provider: str
    search_time_ms: float
    cached: bool = False


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""
    request_id: str = ""