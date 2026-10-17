# Changelog

All notable changes to **Deep Research 104** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [1.0.0] — 2026-10-17

Initial production-grade release of the autonomous research engine
combining web search, reasoning models (OpenAI o-series, DeepSeek-R1),
and inference-time scaling techniques.

### Added

- **Project hygiene**
  - Comprehensive `.gitignore` (Python/IDE/secrets/models), `.dockerignore`,
    `.env.example` documenting 60+ configuration knobs.
  - Migrated `setup.py` to PEP 517 `pyproject.toml` with optional extras
    (`dev`, `training`, `local`, `search`, `observability`, `storage`).
  - Pinned `requirements.txt` + `requirements-dev.txt` with upper bounds.
- **Configuration (`config/`)**
  - `AppSettings` with env-aware nested groups: `ModelConfig`, `SearchConfig`,
    `ReasoningConfig`, `SecurityConfig`, `ObservabilityConfig`, `StorageConfig`.
  - Field validators (temperature/top_p bounds, positive timeouts, capped
    search results, bounded worker count).
  - Cached nested configs via `lru_cache` so HTTP pools aren't re-created.
  - Structured logging via `loguru` with `request_id` / `user_id` / `trace_id`
    contextvars, JSON output in production + rotation, colored output in
    development, separate `errors_*.log` for fast triage.
- **Models (`src/models/`)**
  - `BaseModel` abstract interface with capability advertisement, lifecycle
    hooks (`warmup`, `close`), token validation, per-provider cost
    estimation via `PRICING_TABLE`.
  - `OpenAIModel` (o1 / o3 / o3-mini / gpt-4o) with capability matrix,
    o-series message rewriting, retry policy, request-id capture.
  - `DeepSeekModel` (deepseek-reasoner / chat) with `</tag>` tag
    parsing, fallback reasoning-token estimation, async streaming.
  - `OllamaModel` (deepseek-r1:1.5b..70b, llama3/3.1, mistral, qwen2/2.5)
    with shared `httpx.AsyncClient`, lazy model pull, health-check.
  - `VLLMModel` (thin OpenAI-compatible client) and `LlamaCppModel`
    placeholder for future llama.cpp server integration.
  - `get_model_factory()` dispatch for `openai / deepseek / ollama / vllm / llamacpp`.
- **Reasoning engines (`src/reasoning/`)**
  - `ChainOfThought` with zero-shot / few-shot / structured modes and
    self-consistency (Wang et al., 2022).
  - `TreeOfThoughts` with beam / BFS / DFS search strategies.
  - `ParallelSampler` and `SequentialSampler` for inference-time scaling.
  - `OutcomeRewardModel`, `ProcessRewardModel`, `RuleBasedVerifier`,
    `BestOfNWithVerifier` for verifier-guided search.
  - `InferenceTimeScaler` with `ScalingBudget` and `ScalingStrategy`.
- **Search (`src/search/`)**
  - `FallbackSearchEngine` with provider fallback (DuckDuckGo → SerpAPI →
    Brave → Bing), shared `aiohttp.ClientSession` per provider, retry/backoff,
    URL safety filtering, query validation, namespaced cache keys.
  - `ContentExtractor` with trafilatura → readability → BeautifulSoup cascade,
    optional Playwright for JS-heavy pages, 2 MiB body cap, lazy browser.
  - `SearchAggregator` with multi-factor relevance scoring, domain reputation
    tables (trusted + low-quality), .edu/.gov boosts, defensive URL handling.
- **Pipeline (`src/pipeline/`)**
  - `DeepResearchPipeline` — settings-driven, defensive (every step wrapped
    in try/except), per-iteration metrics (duration / LLM calls / tokens /
    cost), `to_markdown()` + `to_json()` serializers, `request_id` propagation.
  - `MetaCoT` — single-loop reasoning that internalizes search via
    `<search>`, `<reading>`, `<think>` tags (DeepSeek-R1 style).
- **Training (`src/training/`)**
  - `STaRTrainer` (SFT on self-generated reasoning traces).
  - `RewardModelTrainer` (ORM + PRM), `SyntheticPreferenceGenerator`.
  - `GRPOTrainer` (Group-Relative Policy Optimization, RL with verifier).
  - `SelfRefinementEngine` (generate → critique → revise loop).
- **REST API (`src/api/`)**
  - FastAPI app with CORS, GZip, security headers, request body-size cap,
    request-id middleware via contextvars, Prometheus metrics.
  - Endpoints: `POST /research`, `POST /research/stream` (SSE),
    `POST /cot`, `POST /tot`, `POST /search`, `GET /providers`,
    `GET /metrics/cache`, `GET /metrics/search`, `GET /health`,
    `GET /ready`, `GET /version`, `GET /metrics`, `GET /`.
  - Strict Pydantic schemas (`extra='forbid'`, `str_strip_whitespace=True`),
    `ErrorResponse.from_exception()` factory, `HealthResponse` /
    `VersionResponse` / `ReadyResponse`.
  - Production-only startup validation (JWT secret length, at least one
    LLM key, no wildcard CORS).
- **Utils (`src/utils/`)**
  - `TokenCounter` with cached encodings, multimodal (vision) counting,
    prefix fallback for versioned model names.
  - `LayeredCache` (L1 in-process LRU + L2 diskcache + L3 Redis) with
    per-layer metrics, graceful degradation, `clear_pattern`, `get_many`.
  - `InProcessRateLimiter` with race-free per-model locks, burst allowance,
    per-model metrics, search-provider limits.
- **Observability**
  - Prometheus metrics: request count + latency histogram, in-flight gauge,
    token counter, cache hit/miss per layer, search provider success/failure.
  - `/metrics` endpoint exposed without auth (protect at ingress).
  - Optional OpenTelemetry + Sentry hooks via `[observability]` extra.
- **Infrastructure**
  - Multi-stage Dockerfile (base → production slim + development) with
    non-root user, OCI labels, proxy headers, Playwright Chromium.
  - `docker-compose.yml` with env-anchored config, AOF Redis persistence,
    resource limits, log rotation, `--profile local` and `--profile monitoring`.
  - `Makefile` for setup / run / lint / format / test / coverage / security /
    docker / docs.
  - GitHub Actions CI: lint + type-check on Python 3.11, test matrix on
    3.11 + 3.12, Docker build smoke test, pip-audit + Trufflehog security
    scans, codecov upload.
  - Pre-commit hooks: ruff, black, mypy, pip-audit, markdownlint, hadolint.
  - Issue templates (bug, feature), PR template, CODEOWNERS, SECURITY.md,
    CONTRIBUTING.md, Dependabot config.
- **Documentation**
  - `README.md` — full project walkthrough, architecture diagrams,
    quick-start, CLI + Python API references.
  - `Tutorial-104.md` — tutorial-style explanation of each reasoning
    technique and inference-time scaling strategy.

### Security

- Request body-size cap (`REQUEST_MAX_BODY_BYTES`, default 1 MiB).
- Security headers shipped on every response (HSTS in production).
- API keys validated at startup; never logged; never returned in errors.
- Search URLs filtered against `BLOCKED_DOMAINS` + `BLOCKED_EXTENSIONS`
  + scheme allow-list (http/https only).
- Content extractor caps body size at 2 MiB and disables SSL verify only
  for the outbound search session (not the API server).
- `/docs` and `/redoc` disabled in production; `/openapi.json` also disabled.

### Operational

- Liveness probe (`GET /health`) and readiness probe (`GET /ready`, 503
  if any subsystem is down) for Kubernetes.
- Graceful shutdown: closes cache, search engine, extractor, model pools.
- Structured JSON logs in production (enqueue=True, no diagnose).
- Per-request cost accounting surfaced in logs and Prometheus metrics.
