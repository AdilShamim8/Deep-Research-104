# Architecture

> **Build date:** 17 October 2026
> **Maintainer:** Adil Shamim (`@AdilShamim8`)
> **License:** MIT

Deep Research 104 is a production-grade autonomous research engine
that combines web search, reasoning models (OpenAI o-series, DeepSeek-R1,
and local models via Ollama / vLLM), and inference-time scaling
techniques (Chain-of-Thought, Tree of Thoughts, parallel / sequential
sampling, verifier-guided best-of-N, and Meta-CoT).

This document describes the production system design — what the
moving parts are, how they fit together, and where we expect load and
failures to land at 100–200 million active users.

## 1. High-level diagram

```
                            ┌────────────────────────┐
                            │   Ingress / LB / WAF   │
                            │   (TLS, rate limit)     │
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │   FastAPI (uvicorn ×4) │
                            │   src.api.main          │
                            │   ───────────────────  │
                            │   • CORS + GZip         │
                            │   • Security headers    │
                            │   • Body-size cap (1 MiB)│
                            │   • Request-id context  │
                            │   • Prometheus metrics  │
                            └───────────┬────────────┘
                                        │
                ┌───────────────────────┼─────────────────────┐
                │                       │                     │
                ▼                       ▼                     ▼
       ┌──────────────┐         ┌──────────────┐       ┌──────────────┐
       │ DeepResearch │         │   CoT / ToT  │       │   Search     │
       │   Pipeline   │         │   endpoint   │       │   endpoint   │
       └──────┬───────┘         └──────┬───────┘       └──────┬───────┘
              │                         │                      │
              │  ┌──────────────────────┘                      │
              │  │                                              │
              ▼  ▼                                              ▼
       ┌──────────────┐         ┌──────────────┐       ┌──────────────┐
       │   Search     │◀────────│  Reasoning   │       │  Aggregator  │
       │   engine     │         │   engines    │       │   dedup +    │
       │  (DuckDuckGo │         │  (CoT, ToT,  │       │   scoring    │
       │   SerpAPI,   │         │   parallel,  │       └──────────────┘
       │   Brave, Bing│         │   sequential,│
       │   fallback)  │         │   verifier)  │
       └──────┬───────┘         └──────┬───────┘
              │                        │
              ▼                        ▼
       ┌──────────────┐         ┌──────────────┐
       │   Content    │         │    Models    │
       │   Extractor  │         │  (OpenAI /   │
       │ (trafilatura,│         │  DeepSeek /  │
       │  readability,│         │  Ollama /    │
       │  bs4, Playw.)│         │  vLLM)       │
       └──────────────┘         └──────────────┘

  ── Cross-cutting ────────────────────────────────────────────────────────────
  LayeredCache  (LRU + disk + Redis)    →  src.utils.cache
  RateLimiter   (per-model RPM/TPM)     →  src.utils.rate_limiter
  TokenCounter  (tiktoken, multimodal)  →  src.utils.token_counter
  Settings      (Pydantic, env-driven)  →  config.settings
  Logging       (loguru, JSON+contextvars) → config.logging_config
  Metrics       (Prometheus)            →  src.api.main
```

## 2. Module map

| Module | Responsibility | Hardened in 1.0.0 |
|--------|----------------|-------------------|
| `config/` | Env-driven settings + structured logging | Pydantic validators, contextvar correlation, JSON+rotation |
| `src/api/` | FastAPI app, routes, schemas | Security headers, body cap, liveness/readiness, strict schemas |
| `src/models/` | LLM provider clients | Capability matrix, lifecycle hooks, retry, cost tracking |
| `src/search/` | Web search + extraction + aggregation | Shared sessions, URL safety, dedup, domain reputation |
| `src/reasoning/` | CoT / ToT / sampling / verifier | Provider-agnostic, defensive parsing |
| `src/pipeline/` | Deep research + Meta-CoT orchestrators | Settings-driven, defensive, JSON export |
| `src/training/` | STaR / RL / PRM / self-refinement | Documented, optional extras |
| `src/utils/` | Cache, rate limiter, token counter | 3-layer cache, race-free limiter, multimodal counting |
| `tests/` | 80+ unit tests across 6 modules | pytest-asyncio, marks, fixtures, hermetic |

## 3. Production design choices

### 3.1 Why shared `httpx.AsyncClient` per provider

Opening a new HTTP session per request burns TCP connections, TLS
handshakes, and memory. The OpenAI / DeepSeek / Ollama / search
providers each hold a single `httpx.AsyncClient` (or `aiohttp.ClientSession`)
created lazily under a lock and closed cleanly on shutdown. This
reduces p50 latency by ~50ms on every request after the first.

### 3.2 Why three-layer cache (LRU + disk + Redis)

- **L1 (in-process LRU):** ~256 entries, no I/O. Catches the hottest
  repeated queries within a single worker.
- **L2 (diskcache):** per-worker persistent cache (~2 GiB typical).
  Survives worker restarts; survives Redis outages.
- **L3 (Redis):** shared across workers + pods. Source of truth for
  cache invalidation.

Reads cascade L1 → L2 → L3 → None. Writes propagate to all available
layers. If Redis is down, the cache still works via L1+L2 — just no
cross-worker sharing.

### 3.3 Why per-model rate limiting with burst + RPM + TPM

OpenAI, DeepSeek, and search providers each have distinct RPM and TPM
limits. Hitting them costs real money (retries) and kills throughput.
The `InProcessRateLimiter` keeps per-model rolling 60s windows with
asyncio locks; a cross-worker Redis token-bucket is the next step
(see §6 Future work).

### 3.4 Why capability advertisement

Models are heterogeneous. o3-mini supports reasoning but no temperature.
GPT-4o supports vision + tool calls. Local models vary wildly. The
`ModelCapability` set on each `BaseModel` lets callers fail fast via
`require_capability()` instead of sending a malformed request to the
provider and getting a 400 back.

### 3.5 Why contextvar-based correlation IDs

A request that triggers 12 LLM calls + 5 searches should produce 12 +
5 + N log lines all sharing the same `request_id`. We use `contextvars`
(not thread-locals) so async workers don't bleed IDs across requests.
The JSON formatter in production includes `request_id` on every line,
so log aggregators (Loki, ELK, Datadog) can correlate in a single query.

### 3.6 Why `_StrictRequest` (extra='forbid')

A typo like `{"qestion": "..."}` shouldn't silently fall through to
defaults and produce a misleading "question is required" error. The
strict request base rejects unknown fields at validation, returning
a 422 that points at the exact bad field. This saves hours of
integration debugging.

### 3.7 Why defensive pipelines

The Deep Research pipeline wraps every step (query generation, search,
aggregation, gap analysis, synthesis) in try/except. A single bad
iteration (network blip, model 500, malformed response) is logged and
the loop continues with safe fallbacks (empty results, "medium"
confidence). This keeps the API serving useful output even when
upstream providers degrade — crucial at scale.

## 4. Failure modes we've designed for

| Failure | Detection | Mitigation |
|---------|-----------|-----------|
| LLM provider 429 | RateLimitError caught | Exponential backoff (tenacity, max 5 attempts, 4–60s cap) |
| LLM provider 5xx | APIError caught | Same backoff, then HTTP 500 to client with ErrorResponse |
| Redis down | `_get_redis()` returns None | Fall through to disk + LRU; warning logged |
| Disk cache write fails | try/except | Skip; metrics counter incremented |
| Search returns 0 results | `response.has_results == False` | Try next provider; if all fail, return empty response with error |
| Content fetch returns non-HTML | Content-Type sniff | Return None; extractor falls through to next method or reports failure |
| Body too large | `Content-Length` header check | 413 short-circuit before reading body |
| Unknown env var | Pydantic validator | Settings load fails at startup; process exits |
| Missing JWT secret in prod | `_validate_production_requirements` | Logged at startup; /health reports unhealthy |
| Worker crash | Kubernetes liveness probe | Restart pod; cache survives in Redis |
| Bad model name in request | `get_model_factory` ValueError | HTTP 400 to client; no provider call made |

## 5. Observability surface

### Metrics (Prometheus, exposed at `/metrics`)

| Metric | Type | Labels | Source |
|--------|------|--------|--------|
| `deep_research_requests_total` | Counter | method, endpoint, status_code | `src.api.main` |
| `deep_research_request_duration_seconds` | Histogram | endpoint | `src.api.main` |
| `deep_research_tokens_total` | Counter | model, type (prompt/completion/reasoning) | (used by pipeline) |
| `deep_research_inflight_requests` | Gauge | — | `src.api.main` |
| Cache hit/miss/error counts | Python dict | (via `/metrics/cache` endpoint) | `src.utils.cache.CacheMetrics` |
| Search provider requests_made/failed | Python dict | (via `/metrics/search` endpoint) | `src.search.web_search` |

### Logs (loguru, JSON in production)

Every line carries `request_id`, `user_id` (if set), `trace_id` (if
set), plus any `logger.bind(...)` extras. Production sinks:
- stdout (JSON, enqueue=True)
- `logs/app_*.log` (rotated 100 MB / 30 d gz)
- `logs/errors_*.log` (rotated 50 MB / 90 d)

### Tracing (OpenTelemetry, optional)

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to ship spans to your collector.
The `[observability]` extra pulls in
`opentelemetry-instrumentation-fastapi` for auto-instrumentation.

## 6. Future work (not blocking 1.0.0)

- **Redis-backed rate limiter** so workers coordinate on a single
  bucket (today's limiter is per-process).
- **Postgres storage** for research reports (schema designed, not
  wired — see `StorageConfig`).
- **Cost dashboards** in Grafana using `deep_research_tokens_total`.
- **gRPC streaming** for high-throughput clients (SSE works but is
  text-only).
- **Per-tenant quotas** on top of the rate limiter (today, a single
  API key hits the global bucket).
- **Prompt-injection guard** for user-supplied questions before they
  hit the synthesis model.
