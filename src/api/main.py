"""
FastAPI application entry point.
Production-ready with middleware, CORS, metrics, and error handling.
"""

import time
import uuid
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
)

from src.api.routes import router
from src.api.schemas import ErrorResponse
from config.settings import settings
from config.logging_config import setup_logging


# ── Prometheus metrics ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "deep_research_requests_total",
    "Total requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "deep_research_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)
RESEARCH_TOKENS = Counter(
    "deep_research_tokens_total",
    "Total tokens used",
    ["model", "type"],
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    setup_logging()
    logger.info(
        f"Deep Research API starting | env={settings.app_env}"
    )

    # Warm up cache connection
    from src.utils.cache import cache
    try:
        redis = await cache._get_redis()
        if redis:
            logger.info("Cache (Redis) connected")
        else:
            logger.warning("Running without Redis cache")
    except Exception as e:
        logger.warning(f"Cache initialization warning: {e}")

    yield

    # Shutdown
    logger.info("Deep Research API shutting down")
    await cache.close()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Deep Research API",
        description=(
            "Production-grade Deep Research system with web search, "
            "reasoning models (OpenAI o-series, DeepSeek-R1), "
            "and inference-time scaling techniques."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=(
            ["*"] if settings.app_env == "development"
            else ["https://yourdomain.com"]
        ),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Request ID + Logging middleware ─────────────────────────────────────

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        request.state.start_time = time.time()

        logger.info(
            f"[{request_id}] {request.method} {request.url.path}"
        )

        response = await call_next(request)

        latency = time.time() - request.state.start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency:.3f}s"

        # Prometheus metrics
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        logger.info(
            f"[{request_id}] {response.status_code} "
            f"({latency:.3f}s)"
        )
        return response

    # ── Exception handlers ──────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"[{request_id}] Unhandled exception: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc) if settings.app_env != "production" else "",
                request_id=request_id,
            ).model_dump(),
        )

    # ── Routes ──────────────────────────────────────────────────────────────

    app.include_router(router, prefix="/api/v1")

    # Prometheus metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        workers=1 if settings.app_env == "development" else 4,
        log_level=settings.log_level.lower(),
    )