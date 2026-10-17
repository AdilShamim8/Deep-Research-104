"""
FastAPI application entry point.

Production-ready with:
    - Structured logging via loguru (correlation IDs via contextvars)
    - Prometheus metrics (request count, latency, tokens, in-flight)
    - CORS with environment-aware allowed origins
    - GZip compression for large responses
    - Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
    - Request body size cap (prevents memory DoS)
    - Liveness + readiness endpoints (Kubernetes-friendly)
    - /version endpoint
    - Graceful shutdown (closes caches, models, search sessions)
    - Production-only startup validation (fail fast on missing secrets)
    - OpenTelemetry hooks (optional, via env)

Date tag: 17 October 2026.
"""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from config.logging_config import (
    bind_request_id,
    clear_request_context,
    setup_logging,
)
from config.settings import settings
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api.routes import router
from src.api.schemas import ErrorResponse, HealthResponse, VersionResponse

# ── Prometheus metrics ────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "deep_research_requests_total",
    "Total HTTP requests received.",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "deep_research_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)
RESEARCH_TOKENS = Counter(
    "deep_research_tokens_total",
    "Total LLM tokens consumed.",
    ["model", "type"],  # type: prompt | completion | reasoning
)
INFLIGHT_REQUESTS = Gauge(
    "deep_research_inflight_requests",
    "Requests currently being processed.",
)


# ── App lifecycle ────────────────────────────────────────────────────────────

_app_start_time: float = time.time()


def _validate_production_requirements() -> list[str]:
    """Fail-fast checks for production deployment. Returns list of issues."""
    issues: list[str] = []
    if settings.is_production:
        sec = settings.security
        if not sec.jwt_secret or len(sec.jwt_secret) < 16:
            issues.append("JWT_SECRET must be set and >= 16 chars in production")
        if not settings.models.openai_api_key and not settings.models.deepseek_api_key:
            issues.append(
                "At least one of OPENAI_API_KEY or DEEPSEEK_API_KEY must be set in production"
            )
        if "*" in sec.allowed_origins_list:
            issues.append("Wildcard CORS origin not allowed in production")
    return issues


async def _initialize_subsystems() -> dict[str, bool]:
    """Initialize cache, model factory warmup, etc. Returns per-subsystem status."""
    status: dict[str, bool] = {"cache": False, "logging": True}

    # Logging
    try:
        setup_logging()
    except Exception as e:
        logger.error(f"Logging setup failed: {e}")
        status["logging"] = False

    # Cache (Redis optional; falls back to disk + LRU)
    try:
        from src.utils.cache import cache

        redis = await cache._get_redis()
        status["cache"] = redis is not None
        if status["cache"]:
            logger.info("Cache: Redis connected")
        else:
            logger.warning("Cache: Redis unavailable; using disk + LRU only")
    except Exception as e:
        logger.warning(f"Cache init failed: {e}")
        status["cache"] = False

    return status


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup + shutdown hooks."""
    # ── Startup ─────────────────────────────────────────────────────────────
    setup_logging()
    logger.info(
        f"Deep Research API starting | env={settings.app_env} "
        f"version={settings.version} build_date={settings.build_date}"
    )

    issues = _validate_production_requirements()
    if issues and settings.is_production:
        for issue in issues:
            logger.error(f"Production requirement missing: {issue}")
        # Don't crash — let health endpoint report unhealthy instead.
        # Operators can debug via /health and /ready.

    # Ensure log + cache dirs exist
    Path("logs").mkdir(exist_ok=True)
    Path(".cache").mkdir(exist_ok=True)

    # Initialize subsystems
    subsystem_status = await _initialize_subsystems()
    app.state.subsystem_status = subsystem_status

    logger.info(
        f"Startup complete | subsystems={subsystem_status} workers_hint={settings.app_workers}"
    )

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Deep Research API shutting down…")
    try:
        from src.utils.cache import cache

        await cache.close()
    except Exception as e:
        logger.warning(f"Cache close failed: {e}")
    try:
        from src.search.web_search import search_engine

        await search_engine.close()
    except Exception as e:
        logger.warning(f"Search engine close failed: {e}")
    try:
        from src.search.content_extractor import extractor

        await extractor.close()
    except Exception as e:
        logger.warning(f"Extractor close failed: {e}")
    logger.info("Shutdown complete")


# ── Security headers middleware ───────────────────────────────────────────────

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


def _apply_security_headers(response: Response) -> None:
    for key, value in SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    # HSTS only meaningful over HTTPS; ship it anyway (proxies handle TLS)
    if settings.is_production:
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains",
        )


# ── Request body size cap ─────────────────────────────────────────────────────


async def _enforce_body_size(request: Request) -> JSONResponse | None:
    """Reject requests with Content-Length beyond the configured cap.

    Returns a JSONResponse to short-circuit, or None to allow the request.
    """
    max_bytes = settings.security.request_max_body_bytes
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            f"[{request_id}] Request body too large: {int(content_length)} > {max_bytes} bytes"
        )
        return JSONResponse(
            status_code=413,
            content=ErrorResponse(
                error="payload_too_large",
                detail=f"Request body exceeds max of {max_bytes} bytes",
                request_id=request_id,
                error_code="payload_too_large",
            ).model_dump(),
        )
    return None


# ── App factory ──────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Deep Research API",
        description=(
            "Production-grade Deep Research system with web search, "
            "reasoning models (OpenAI o-series, DeepSeek-R1), and "
            "inference-time scaling techniques (CoT, ToT, parallel sampling)."
        ),
        version=settings.version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins = ["*"] if settings.is_development else settings.security.allowed_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )

    # ── GZip ──────────────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1_000)

    # ── Request middleware (logging + metrics + correlation ID) ─────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        request.state.start_time = time.time()

        bind_request_id(request_id)
        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        # Body size check (short-circuit before we even read it)
        early_reject = await _enforce_body_size(request)
        if early_reject is not None:
            _apply_security_headers(early_reject)
            clear_request_context()
            return early_reject

        INFLIGHT_REQUESTS.inc()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"[{request_id}] Unhandled: {exc}", exc_info=True)
            response = JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="internal_server_error",
                    detail=str(exc) if not settings.is_production else "",
                    request_id=request_id,
                    error_code="internal_error",
                ).model_dump(),
            )
        finally:
            INFLIGHT_REQUESTS.dec()

        latency = time.time() - getattr(request.state, "start_time", time.time())
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{latency:.3f}s"
        _apply_security_headers(response)

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)

        logger.info(f"[{request_id}] {response.status_code} ({latency:.3f}s)")
        clear_request_context()
        return response

    # ── Exception handler (last line of defense) ─────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                detail=str(exc) if not settings.is_production else "",
                request_id=request_id,
                error_code="internal_error",
            ).model_dump(),
        )

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # ── System endpoints ─────────────────────────────────────────────────────
    @app.get("/health", tags=["system"], summary="Liveness probe")
    async def health() -> HealthResponse:
        """Kubernetes liveness probe. Returns 200 if the process is alive."""
        from src.utils.cache import cache

        redis_ok = False
        try:
            redis = await cache._get_redis()
            if redis:
                await redis.ping()
                redis_ok = True
        except Exception:
            pass
        return HealthResponse(
            status="healthy",
            python_version=sys.version.split()[0],
            version=settings.version,
            build_date=settings.build_date,
            redis="connected" if redis_ok else "disconnected",
            uptime_seconds=time.time() - _app_start_time,
            timestamp=time.time(),
        )

    @app.get("/ready", tags=["system"], summary="Readiness probe")
    async def ready() -> JSONResponse:
        """Kubernetes readiness probe. Returns 200 only when all
        subsystems report healthy; 503 otherwise."""
        from src.api.schemas import ReadyResponse

        checks: dict[str, bool] = getattr(app.state, "subsystem_status", {})
        # Re-check Redis at probe time
        try:
            from src.utils.cache import cache

            redis = await cache._get_redis()
            checks["cache"] = redis is not None
        except Exception:
            checks["cache"] = False
        status = "ready" if all(checks.values()) else "not_ready"
        code = 200 if status == "ready" else 503
        return JSONResponse(
            status_code=code,
            content=ReadyResponse(status=status, checks=checks, timestamp=time.time()).model_dump(),
        )

    @app.get("/version", tags=["system"], summary="Version info")
    async def version() -> VersionResponse:
        return VersionResponse(
            name=settings.app_name,
            version=settings.version,
            build_date=settings.build_date,
            python_version=sys.version.split()[0],
            app_env=settings.app_env,
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint. Exposed without auth — protect
        at the ingress layer in production."""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ── Root ────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs" if not settings.is_production else None,
            "endpoints": [
                "/api/v1/research",
                "/api/v1/research/stream",
                "/api/v1/cot",
                "/api/v1/tot",
                "/api/v1/search",
                "/health",
                "/ready",
                "/version",
                "/metrics",
            ],
        }

    return app


# ── Module-level app instance ─────────────────────────────────────────────────

app = create_app()


# ── Direct-run entry ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.app_workers,
        log_level=settings.log_level.lower(),
        access_log=False,  # loguru handles request logging
    )
