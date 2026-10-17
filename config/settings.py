"""
Central configuration management using Pydantic Settings.

Loads from environment variables (and optional .env file) with full
validation, type coercion, and environment-aware defaults. Designed
for 12-factor app deployments across development / staging / production.

Key design decisions:
- Nested configs are exposed as cached properties (single instance each).
- Production defaults fail-safe: empty secrets raise on startup, not later.
- All time/buffer values are tunable via env without code changes.
- Date tag: 17 October 2026.

Author: Adil Shamim <adilshamim8@users.noreply.github.com>
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Model providers ────────────────────────────────────────────────────────────

class ModelConfig(BaseSettings):
    """Configuration for individual LLM providers."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # OpenAI / o-family
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_org_id: Optional[str] = Field(default=None, env="OPENAI_ORG_ID")
    openai_default_model: str = Field(default="o3-mini", env="OPENAI_DEFAULT_MODEL")
    openai_reasoning_model: str = Field(default="o3", env="OPENAI_REASONING_MODEL")
    openai_reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium", env="OPENAI_REASONING_EFFORT"
    )
    openai_request_timeout: int = Field(default=120, env="OPENAI_REQUEST_TIMEOUT")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")

    # DeepSeek
    deepseek_api_key: str = Field(default="", env="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1", env="DEEPSEEK_BASE_URL"
    )
    deepseek_reasoning_model: str = Field(
        default="deepseek-reasoner", env="DEEPSEEK_REASONING_MODEL"
    )
    deepseek_request_timeout: int = Field(default=180, env="DEEPSEEK_REQUEST_TIMEOUT")
    deepseek_max_retries: int = Field(default=3, env="DEEPSEEK_MAX_RETRIES")

    # Local deployment
    local_model_path: str = Field(default="./models", env="LOCAL_MODEL_PATH")
    local_model_type: Literal["ollama", "vllm", "llamacpp"] = Field(
        default="ollama", env="LOCAL_MODEL_TYPE"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", env="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="deepseek-r1:7b", env="OLLAMA_MODEL")

    @field_validator("openai_api_key", "deepseek_api_key")
    @classmethod
    def _strip_key(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("openai_request_timeout", "deepseek_request_timeout")
    @classmethod
    def _positive_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("request timeout must be positive")
        return v


# ── Web search ─────────────────────────────────────────────────────────────────

class SearchConfig(BaseSettings):
    """Search engine configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    serpapi_key: str = Field(default="", env="SERPAPI_KEY")
    brave_search_api_key: str = Field(default="", env="BRAVE_SEARCH_API_KEY")
    bing_search_api_key: str = Field(default="", env="BING_SEARCH_API_KEY")

    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    search_timeout_seconds: int = Field(default=15, env="SEARCH_TIMEOUT_SECONDS")
    max_content_length: int = Field(default=8000, env="MAX_CONTENT_LENGTH")

    # Search engine priority
    search_provider: Literal["serpapi", "brave", "bing", "duckduckgo"] = Field(
        default="duckduckgo", env="SEARCH_PROVIDER"
    )

    @field_validator("max_search_results")
    @classmethod
    def _bound_results(cls, v: int) -> int:
        if not 1 <= v <= 50:
            raise ValueError("max_search_results must be between 1 and 50")
        return v

    @field_validator("search_timeout_seconds")
    @classmethod
    def _bound_timeout(cls, v: int) -> int:
        if not 1 <= v <= 120:
            raise ValueError("search_timeout_seconds must be between 1 and 120")
        return v


# ── Reasoning pipeline ───────────────────────────────────────────────────────

class ReasoningConfig(BaseSettings):
    """Reasoning pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Token budgets
    max_tokens_per_request: int = Field(
        default=128_000, env="MAX_TOKENS_PER_REQUEST"
    )
    reasoning_budget: int = Field(default=32_768, env="REASONING_BUDGET")
    max_output_tokens: int = Field(default=8_192, env="MAX_OUTPUT_TOKENS")

    # Sampling
    parallel_samples: int = Field(default=3, env="PARALLEL_SAMPLES")
    sequential_steps: int = Field(default=5, env="SEQUENTIAL_STEPS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")

    # Tree of Thoughts
    tot_branching_factor: int = Field(default=3, env="TOT_BRANCHING_FACTOR")
    tot_max_depth: int = Field(default=4, env="TOT_MAX_DEPTH")
    tot_beam_width: int = Field(default=2, env="TOT_BEAM_WIDTH")

    # Research loop
    max_research_iterations: int = Field(
        default=5, env="MAX_RESEARCH_ITERATIONS"
    )
    max_search_per_iteration: int = Field(
        default=3, env="MAX_SEARCH_PER_ITERATION"
    )

    @field_validator("temperature")
    @classmethod
    def _bound_temp(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        return v

    @field_validator("top_p")
    @classmethod
    def _bound_top_p(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("top_p must be in (0.0, 1.0]")
        return v

    @field_validator("parallel_samples", "sequential_steps")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


# ── Security ─────────────────────────────────────────────────────────────────

class SecurityConfig(BaseSettings):
    """API gateway / authentication / rate limiting."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    api_key_header: str = Field(default="x-api-key", env="API_KEY_HEADER")
    jwt_secret: str = Field(default="", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=60, env="JWT_EXPIRE_MINUTES")
    allowed_origins: str = Field(
        default="https://deepresearch.space-z.ai", env="ALLOWED_ORIGINS"
    )
    rate_limit_rpm: int = Field(default=60, env="RATE_LIMIT_RPM")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")
    request_max_body_bytes: int = Field(
        default=1_048_576, env="REQUEST_MAX_BODY_BYTES"
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @field_validator("jwt_secret")
    @classmethod
    def _warn_empty_secret(cls, v: str) -> str:
        # Don't raise in dev (allows empty); production check happens at startup.
        return v

    @field_validator("rate_limit_rpm")
    @classmethod
    def _positive_rpm(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("rate_limit_rpm must be positive")
        return v


# ── Observability ────────────────────────────────────────────────────────────

class ObservabilityConfig(BaseSettings):
    """Telemetry, tracing, metrics."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    otel_exporter_otlp_endpoint: str = Field(
        default="", env="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_service_name: str = Field(
        default="deep-research-104", env="OTEL_SERVICE_NAME"
    )
    sentry_dsn: str = Field(default="", env="SENTRY_DSN")


# ── Storage ──────────────────────────────────────────────────────────────────

class StorageConfig(BaseSettings):
    """Persistent storage for research results."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str = Field(
        default="postgresql+asyncpg://deep:deep@localhost:5432/deep_research",
        env="DATABASE_URL",
    )
    storage_backend: Literal["postgres", "sqlite", "memory"] = Field(
        default="memory", env="STORAGE_BACKEND"
    )


# ── Master settings ──────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    """Master application settings — singleton at module import."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App identity
    app_name: str = Field(default="deep-research-104", env="APP_NAME")
    app_env: Literal["development", "staging", "production", "test"] = Field(
        default="development", env="APP_ENV"
    )
    app_port: int = Field(default=8000, env="APP_PORT")
    app_workers: int = Field(default=4, env="APP_WORKERS")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", env="LOG_LEVEL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", env="REDIS_URL"
    )
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")

    # Build / release metadata
    build_date: str = Field(default="2026-10-17", env="BUILD_DATE")
    version: str = Field(default="1.0.0", env="APP_VERSION")

    # Nested configs — cached via lru_cache below
    @property
    def models(self) -> ModelConfig:
        return _get_model_config()

    @property
    def search(self) -> SearchConfig:
        return _get_search_config()

    @property
    def reasoning(self) -> ReasoningConfig:
        return _get_reasoning_config()

    @property
    def security(self) -> SecurityConfig:
        return _get_security_config()

    @property
    def observability(self) -> ObservabilityConfig:
        return _get_observability_config()

    @property
    def storage(self) -> StorageConfig:
        return _get_storage_config()

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("app_env")
    @classmethod
    def _lower_env(cls, v: str) -> str:
        return v.lower()

    @field_validator("app_workers")
    @classmethod
    def _positive_workers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("app_workers must be positive")
        return v

    @model_validator(mode="after")
    def _enforce_production_constraints(self) -> "AppSettings":
        """Production-only hard failures."""
        if self.app_env == "production":
            if not self.security.jwt_secret if hasattr(self, "security") else False:
                # jwt_secret lives on SecurityConfig; we cannot reach it cleanly
                # from here, so the actual production check is enforced in
                # src.api.main lifespan startup. Keep this hook for future use.
                pass
        return self

    # ── Convenience flags ─────────────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_staging(self) -> bool:
        return self.app_env == "staging"

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"


# ── Cached nested config accessors ────────────────────────────────────────────
# Using lru_cache ensures a single shared instance per process, which keeps
# Redis connections / HTTP pools cheap and avoids surprising re-instantiation.

@lru_cache(maxsize=1)
def _get_model_config() -> ModelConfig:
    return ModelConfig()

@lru_cache(maxsize=1)
def _get_search_config() -> SearchConfig:
    return SearchConfig()

@lru_cache(maxsize=1)
def _get_reasoning_config() -> ReasoningConfig:
    return ReasoningConfig()

@lru_cache(maxsize=1)
def _get_security_config() -> SecurityConfig:
    return SecurityConfig()

@lru_cache(maxsize=1)
def _get_observability_config() -> ObservabilityConfig:
    return ObservabilityConfig()

@lru_cache(maxsize=1)
def _get_storage_config() -> StorageConfig:
    return StorageConfig()


# ── Singleton accessor ────────────────────────────────────────────────────────
# Other modules import `settings` from here. Do NOT instantiate AppSettings
# elsewhere — that would bypass env validation and caching.

settings = AppSettings()
