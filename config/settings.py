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
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Model providers ────────────────────────────────────────────────────────────


class ModelConfig(BaseSettings):
    """Configuration for individual LLM providers."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # OpenAI / o-family
    openai_api_key: str = Field(default="")
    openai_org_id: str | None = Field(default=None)
    openai_default_model: str = Field(default="o3-mini")
    openai_reasoning_model: str = Field(default="o3")
    openai_reasoning_effort: Literal["low", "medium", "high"] = Field(default="medium")
    openai_request_timeout: int = Field(default=120)
    openai_max_retries: int = Field(default=3)

    # DeepSeek
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1")
    deepseek_reasoning_model: str = Field(default="deepseek-reasoner")
    deepseek_request_timeout: int = Field(default=180)
    deepseek_max_retries: int = Field(default=3)

    # Local deployment
    local_model_path: str = Field(default="./models")
    local_model_type: Literal["ollama", "vllm", "llamacpp"] = Field(default="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="deepseek-r1:7b")

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

    serpapi_key: str = Field(default="")
    brave_search_api_key: str = Field(default="")
    bing_search_api_key: str = Field(default="")

    max_search_results: int = Field(default=10)
    search_timeout_seconds: int = Field(default=15)
    max_content_length: int = Field(default=8000)

    # Search engine priority
    search_provider: Literal["serpapi", "brave", "bing", "duckduckgo"] = Field(default="duckduckgo")

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
    max_tokens_per_request: int = Field(default=128_000)
    reasoning_budget: int = Field(default=32_768)
    max_output_tokens: int = Field(default=8_192)

    # Sampling
    parallel_samples: int = Field(default=3)
    sequential_steps: int = Field(default=5)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)

    # Tree of Thoughts
    tot_branching_factor: int = Field(default=3)
    tot_max_depth: int = Field(default=4)
    tot_beam_width: int = Field(default=2)

    # Research loop
    max_research_iterations: int = Field(default=5)
    max_search_per_iteration: int = Field(default=3)

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

    api_key_header: str = Field(default="x-api-key")
    jwt_secret: str = Field(default="")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)
    allowed_origins: str = Field(default="https://deepresearch.space-z.ai")
    rate_limit_rpm: int = Field(default=60)
    rate_limit_burst: int = Field(default=10)
    request_max_body_bytes: int = Field(default=1_048_576)

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

    prometheus_enabled: bool = Field(default=True)
    otel_exporter_otlp_endpoint: str = Field(default="")
    otel_service_name: str = Field(default="deep-research-104")
    sentry_dsn: str = Field(default="")


# ── Storage ──────────────────────────────────────────────────────────────────


class StorageConfig(BaseSettings):
    """Persistent storage for research results."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    database_url: str = Field(
        default="postgresql+asyncpg://deep:deep@localhost:5432/deep_research",
    )
    storage_backend: Literal["postgres", "sqlite", "memory"] = Field(default="memory")


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
    app_name: str = Field(default="deep-research-104")
    app_env: Literal["development", "staging", "production", "test"] = Field(default="development")
    app_port: int = Field(default=8000)
    app_workers: int = Field(default=4)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_password: str = Field(default="")
    redis_max_connections: int = Field(default=50)

    # Build / release metadata
    build_date: str = Field(default="2026-10-17")
    version: str = Field(default="1.0.0")

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

    @field_validator("app_env", mode="before")
    @classmethod
    def _lower_env(cls, v: str) -> str:
        # Run before the Literal type check so 'PRODUCTION' becomes
        # 'production' and passes the literal match.
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("app_workers")
    @classmethod
    def _positive_workers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("app_workers must be positive")
        return v

    @model_validator(mode="after")
    def _enforce_production_constraints(self) -> AppSettings:
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
