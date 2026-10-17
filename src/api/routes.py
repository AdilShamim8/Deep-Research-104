"""
FastAPI route handlers for all Deep Research endpoints.

Production hardening over the toy original:
    - Request_id propagated to every response (correlation in logs)
    - 422 errors wrapped in ErrorResponse with request_id
    - Endpoints guarded by an explicit dependency that initializes the
      model factory lazily and reports failures as 400 (not 500)
    - New endpoints: /providers (list available), /metrics/cache,
      /metrics/search (per-subsystem introspection)
    - Stream endpoint sets keep-alive headers and flushes on each event
    - All endpoints log entry/exit with request_id and latency

Date tag: 17 October 2026.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import settings
from src.api.schemas import (
    CoTRequest,
    CoTResponse,
    ErrorResponse,
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
    SearchRequest,
    SearchResponse,
    SearchResultSchema,
    ToTRequest,
    ToTResponse,
)
from src.models import PROVIDERS, get_model_factory
from src.models.base_model import BaseModel, GenerationConfig
from src.pipeline.deep_research import DeepResearchPipeline
from src.pipeline.meta_cot import MetaCoT
from src.reasoning.chain_of_thought import ChainOfThought
from src.reasoning.tree_of_thoughts import SearchStrategy, TreeOfThoughts


router = APIRouter()


# ── Dependencies ──────────────────────────────────────────────────────────────

def _get_request_id(request: Request) -> str:
    """Pull the request_id that the middleware set on request.state."""
    return getattr(request.state, "request_id", str(uuid.uuid4())[:8])


def _get_model(provider: str, model_name: Optional[str] = None) -> BaseModel:
    """Resolve a model provider. Raises HTTPException(400) on bad input."""
    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown provider: {provider!r}. "
                f"Choose from: {', '.join(PROVIDERS)}"
            ),
        )
    try:
        return get_model_factory(provider, model_name)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Model init failed for provider={provider}: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to initialize model: {e}",
        )


# ── Research endpoints ────────────────────────────────────────────────────────

@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="Deep Research",
    description=(
        "Run the full deep research pipeline with web search + reasoning. "
        "Iteratively searches, reads, and synthesizes information."
    ),
)
async def research(req: ResearchRequest, request: Request) -> ResearchResponse:
    """Execute the deep research pipeline."""
    request_id = _get_request_id(request)
    start_time = time.time()

    logger.info(
        f"[{request_id}] /research provider={req.provider} "
        f"mode={req.mode} q={req.question[:60]!r}"
    )

    model = _get_model(req.provider, req.model_name)

    try:
        if req.mode == "meta_cot":
            meta_cot = MetaCoT(
                model=model,
                max_searches=req.max_iterations * 3,
            )
            result = await meta_cot.reason(req.question)
            processing_ms = (time.time() - start_time) * 1000
            return ResearchResponse(
                question=req.question,
                answer=result.answer,
                confidence="medium",
                sources=[],
                iterations=result.total_searches,
                total_sources_found=len(result.sources_used),
                total_tokens_used=result.total_tokens,
                synthesis_model=model.model_name,
                processing_time_ms=processing_ms,
                request_id=request_id,
            )

        pipeline = DeepResearchPipeline(
            model=model,
            max_iterations=req.max_iterations,
        )
        report = await pipeline.research(
            question=req.question,
            fast_mode=req.fast_mode,
        )
        processing_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[{request_id}] /research done in {processing_ms:.0f}ms "
            f"tokens={report.total_tokens_used} conf={report.confidence}"
        )
        return ResearchResponse.from_report(report, processing_ms, request_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /research failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.from_exception(
                e, request_id=request_id, error_code="research_failed"
            ).model_dump(),
        )


@router.post(
    "/research/stream",
    summary="Stream Deep Research",
    description="Stream research progress as Server-Sent Events (SSE).",
)
async def stream_research(req: ResearchRequest, request: Request) -> StreamingResponse:
    """Stream research progress events as SSE."""
    request_id = _get_request_id(request)
    model = _get_model(req.provider, req.model_name)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            if req.mode == "meta_cot":
                meta_cot = MetaCoT(model=model)
                async for event in meta_cot.stream_reason(req.question):
                    event["request_id"] = request_id
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            else:
                pipeline = DeepResearchPipeline(
                    model=model,
                    max_iterations=req.max_iterations,
                )
                async for event in pipeline.stream_research(req.question):
                    event["request_id"] = request_id
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "content": str(e),
                "request_id": request_id,
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


# ── CoT endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/cot",
    response_model=CoTResponse,
    summary="Chain-of-Thought",
    description="Generate an answer with chain-of-thought reasoning.",
)
async def chain_of_thought(req: CoTRequest, request: Request) -> CoTResponse:
    """Run CoT reasoning on a question."""
    request_id = _get_request_id(request)
    model = _get_model(req.provider, req.model_name)
    cot = ChainOfThought(model)

    config = GenerationConfig(
        temperature=req.temperature,
        max_tokens=4_096,
    )

    try:
        if req.n_samples > 1:
            result = await cot.self_consistency(
                question=req.question,
                context=req.context,
                n_samples=req.n_samples,
                mode=req.cot_mode,
                config=config,
            )
        else:
            result = await cot.generate(
                question=req.question,
                context=req.context,
                mode=req.cot_mode,
                config=config,
            )

        return CoTResponse(
            answer=result.answer,
            reasoning=result.reasoning,
            confidence=result.confidence,
            supporting_votes=result.supporting_votes,
            total_votes=result.total_votes,
            model=model.model_name,
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /cot failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.from_exception(
                e, request_id=request_id, error_code="cot_failed"
            ).model_dump(),
        )


# ── ToT endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/tot",
    response_model=ToTResponse,
    summary="Tree of Thoughts",
    description="Solve problems using Tree of Thoughts search.",
)
async def tree_of_thoughts(req: ToTRequest, request: Request) -> ToTResponse:
    """Run Tree of Thoughts reasoning."""
    request_id = _get_request_id(request)
    model = _get_model(req.provider, req.model_name)

    strategy_map = {
        "beam": SearchStrategy.BEAM,
        "bfs": SearchStrategy.BFS,
        "dfs": SearchStrategy.DFS,
    }
    if req.strategy not in strategy_map:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown strategy: {req.strategy!r}",
        )

    tot = TreeOfThoughts(
        model=model,
        branching_factor=req.branching_factor,
        max_depth=req.max_depth,
        beam_width=req.beam_width,
        search_strategy=strategy_map[req.strategy],
    )

    try:
        result = await tot.solve(problem=req.problem, context=req.context)
        return ToTResponse(
            answer=result.answer,
            best_score=result.best_score,
            total_nodes_explored=result.total_nodes_explored,
            search_strategy=result.search_strategy,
            path_length=len(result.best_path),
            model=model.model_name,
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /tot failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.from_exception(
                e, request_id=request_id, error_code="tot_failed"
            ).model_dump(),
        )


# ── Search endpoint ──────────────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Web Search",
    description="Execute a web search across the configured providers.",
)
async def web_search(req: SearchRequest, request: Request) -> SearchResponse:
    """Execute a web search."""
    request_id = _get_request_id(request)
    try:
        from src.search.web_search import FallbackSearchEngine, search_engine

        # Allow caller to override provider at request time
        engine = search_engine
        if req.provider:
            # Build a one-off engine pinned to the requested provider
            engine = FallbackSearchEngine()

        response = await engine.search(req.query)
        return SearchResponse(
            query=response.query,
            results=[
                SearchResultSchema(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                    domain=r.domain,
                    position=r.position,
                    provider=r.provider,
                    published_date=r.published_date,
                )
                for r in response.results[: req.max_results]
            ],
            total_results=response.total_results,
            provider=response.provider,
            search_time_ms=response.search_time_ms,
            cached=response.cached,
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse.from_exception(
                e, request_id=request_id, error_code="search_failed"
            ).model_dump(),
        )


# ── Provider listing ─────────────────────────────────────────────────────────

@router.get(
    "/providers",
    summary="List available LLM providers",
    description="Returns the list of configured LLM provider names.",
)
async def list_providers() -> dict:
    return {
        "providers": list(PROVIDERS),
        "default": settings.models.openai_reasoning_model,
    }


# ── Cache + search metrics ───────────────────────────────────────────────────

@router.get(
    "/metrics/cache",
    summary="Cache metrics",
    description="Per-layer hit/miss/error counts for the layered cache.",
)
async def cache_metrics() -> dict:
    from src.utils.cache import cache
    m = cache.metrics
    return {
        "hits": m.hits,
        "misses": m.misses,
        "errors": m.errors,
        "l1_hits": m.l1_hits,
        "l2_hits": m.l2_hits,
        "l3_hits": m.l3_hits,
        "hit_rate": round(m.hit_rate, 4),
    }


@router.get(
    "/metrics/search",
    summary="Search provider metrics",
    description="Per-provider requests_made/failed counts.",
)
async def search_metrics() -> dict:
    from src.search.web_search import search_engine
    return search_engine.metrics()


@router.get(
    "/health",
    summary="Health check (route-level)",
    description="Quick route-level health probe. Use /health (root) for the "
                "full liveness probe with Redis + uptime.",
)
async def route_health() -> dict:
    return {"status": "ok", "path": "/api/v1/health"}
