"""
FastAPI route handlers for all endpoints.
"""

import time
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
from loguru import logger

from src.api.schemas import (
    ResearchRequest, ResearchResponse,
    CoTRequest, CoTResponse,
    ToTRequest, ToTResponse,
    SearchRequest, SearchResponse,
    SearchResultSchema, ErrorResponse,
)
from src.models.local_models import get_model_factory
from src.reasoning.chain_of_thought import ChainOfThought
from src.reasoning.tree_of_thoughts import TreeOfThoughts, SearchStrategy
from src.reasoning.parallel_sampling import ParallelSampler
from src.pipeline.deep_research import DeepResearchPipeline
from src.pipeline.meta_cot import MetaCoT
from src.search.web_search import search_engine
from src.models.base_model import GenerationConfig


router = APIRouter()


def _get_model(provider: str, model_name=None):
    """Get model instance from provider string."""
    try:
        return get_model_factory(provider, model_name)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to initialize model: {e}"
        )


# ── Research endpoints ────────────────────────────────────────────────────────

@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="Deep Research",
    description=(
        "Run full deep research pipeline with web search and reasoning. "
        "Iteratively searches, reads, and synthesizes information."
    ),
)
async def research(req: ResearchRequest) -> ResearchResponse:
    """Execute deep research pipeline."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        f"[{request_id}] Research request | "
        f"provider={req.provider} | mode={req.mode} | "
        f"question={req.question[:60]}..."
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

            # Wrap MetaCoT result in ResearchResponse format
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
            )

        else:
            # Default: deep research pipeline
            pipeline = DeepResearchPipeline(
                model=model,
                max_iterations=req.max_iterations,
            )
            report = await pipeline.research(
                question=req.question,
                fast_mode=req.fast_mode,
            )
            processing_ms = (time.time() - start_time) * 1000
            return ResearchResponse.from_report(report, processing_ms)

    except Exception as e:
        logger.error(f"[{request_id}] Research failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/research/stream",
    summary="Stream Deep Research",
    description="Stream research progress as Server-Sent Events.",
)
async def stream_research(req: ResearchRequest) -> StreamingResponse:
    """Stream research progress events."""
    model = _get_model(req.provider, req.model_name)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            if req.mode == "meta_cot":
                meta_cot = MetaCoT(model=model)
                async for event in meta_cot.stream_reason(req.question):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                pipeline = DeepResearchPipeline(
                    model=model,
                    max_iterations=req.max_iterations,
                )
                async for event in pipeline.stream_research(req.question):
                    yield f"data: {json.dumps(event)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_event = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── CoT endpoint ──────────────────────────────────────────────────────────────

@router.post(
    "/cot",
    response_model=CoTResponse,
    summary="Chain-of-Thought",
    description="Generate answer with chain-of-thought reasoning.",
)
async def chain_of_thought(req: CoTRequest) -> CoTResponse:
    """Run CoT reasoning on a question."""
    model = _get_model(req.provider, req.model_name)
    cot = ChainOfThought(model)

    config = GenerationConfig(
        temperature=req.temperature,
        max_tokens=4096,
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
        )
    except Exception as e:
        logger.error(f"CoT failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── ToT endpoint ──────────────────────────────────────────────────────────────

@router.post(
    "/tot",
    response_model=ToTResponse,
    summary="Tree of Thoughts",
    description="Solve problems using Tree of Thoughts search.",
)
async def tree_of_thoughts(req: ToTRequest) -> ToTResponse:
    """Run Tree of Thoughts reasoning."""
    model = _get_model(req.provider, req.model_name)

    strategy_map = {
        "beam": SearchStrategy.BEAM,
        "bfs": SearchStrategy.BFS,
        "dfs": SearchStrategy.DFS,
    }

    tot = TreeOfThoughts(
        model=model,
        branching_factor=req.branching_factor,
        max_depth=req.max_depth,
        beam_width=req.beam_width,
        search_strategy=strategy_map[req.strategy],
    )

    try:
        result = await tot.solve(
            problem=req.problem,
            context=req.context,
        )
        return ToTResponse(
            answer=result.answer,
            best_score=result.best_score,
            total_nodes_explored=result.total_nodes_explored,
            search_strategy=result.search_strategy,
            path_length=len(result.best_path),
            model=model.model_name,
        )
    except Exception as e:
        logger.error(f"ToT failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Search endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Web Search",
    description="Execute web search across configured providers.",
)
async def web_search(req: SearchRequest) -> SearchResponse:
    """Execute web search."""
    try:
        from src.search.web_search import FallbackSearchEngine
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
                )
                for r in response.results[:req.max_results]
            ],
            total_results=response.total_results,
            provider=response.provider,
            search_time_ms=response.search_time_ms,
            cached=response.cached,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ──────────────────────────────────────────────────────────────

@router.get("/health", summary="Health Check")
async def health() -> dict:
    """Service health check."""
    from src.utils.cache import cache
    import sys

    redis_ok = False
    try:
        redis = await cache._get_redis()
        if redis:
            await redis.ping()
            redis_ok = True
    except Exception:
        pass

    return {
        "status": "healthy",
        "python_version": sys.version.split()[0],
        "redis": "connected" if redis_ok else "disconnected",
        "timestamp": time.time(),
    }