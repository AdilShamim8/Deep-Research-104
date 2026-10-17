"""
Unit tests for config.settings.

Verifies env-driven defaults, field validators, environment parsing,
and the cached nested-config accessors (ModelConfig, SearchConfig,
ReasoningConfig, SecurityConfig).
"""

from __future__ import annotations

import os

import pytest

from config.settings import (
    AppSettings,
    ModelConfig,
    ReasoningConfig,
    SearchConfig,
    SecurityConfig,
)


# ── Default values ────────────────────────────────────────────────────────────

class TestDefaults:
    def test_app_env_default_is_development(self) -> None:
        s = AppSettings()
        assert s.app_env == "development"

    def test_log_level_default_is_info(self) -> None:
        s = AppSettings()
        assert s.log_level == "INFO"

    def test_app_port_default(self) -> None:
        s = AppSettings()
        assert s.app_port == 8000

    def test_app_workers_default(self) -> None:
        s = AppSettings()
        assert s.app_workers == 4

    def test_build_date_default(self) -> None:
        s = AppSettings()
        assert s.build_date == "2026-10-17"

    def test_version_default(self) -> None:
        s = AppSettings()
        assert s.version == "1.0.0"


# ── Env-var driven ────────────────────────────────────────────────────────────

class TestEnvDriven:
    def test_app_env_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "PRODUCTION")
        s = AppSettings()
        assert s.app_env == "production"

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = AppSettings()
        assert s.log_level == "DEBUG"

    def test_redis_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://custom:6379/3")
        s = AppSettings()
        assert "custom:6379" in s.redis_url

    def test_app_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_PORT", "9000")
        s = AppSettings()
        assert s.app_port == 9000


# ── Convenience flags ─────────────────────────────────────────────────────────

class TestFlags:
    def test_is_development_true_in_dev(self) -> None:
        s = AppSettings()
        assert s.is_development
        assert not s.is_production
        assert not s.is_staging

    def test_is_production_true_in_prod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        s = AppSettings()
        assert s.is_production
        assert not s.is_development
        assert not s.is_staging

    def test_is_staging_true_in_staging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "staging")
        s = AppSettings()
        assert s.is_staging
        assert not s.is_production


# ── Validators ──────────────────────────────────────────────────────────────────

class TestModelConfigValidation:
    def test_openai_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ModelConfig(openai_request_timeout=0)

    def test_api_key_stripped(self) -> None:
        c = ModelConfig(openai_api_key="  sk-x  ")
        assert c.openai_api_key == "sk-x"


class TestSearchConfigValidation:
    def test_max_results_bounds(self) -> None:
        with pytest.raises(Exception):
            SearchConfig(max_search_results=0)
        with pytest.raises(Exception):
            SearchConfig(max_search_results=100)

    def test_timeout_bounds(self) -> None:
        with pytest.raises(Exception):
            SearchConfig(search_timeout_seconds=0)
        with pytest.raises(Exception):
            SearchConfig(search_timeout_seconds=200)


class TestReasoningConfigValidation:
    def test_temperature_bounds(self) -> None:
        with pytest.raises(Exception):
            ReasoningConfig(temperature=-0.1)
        with pytest.raises(Exception):
            ReasoningConfig(temperature=2.5)

    def test_top_p_bounds(self) -> None:
        with pytest.raises(Exception):
            ReasoningConfig(top_p=0.0)
        with pytest.raises(Exception):
            ReasoningConfig(top_p=1.5)

    def test_parallel_samples_positive(self) -> None:
        with pytest.raises(Exception):
            ReasoningConfig(parallel_samples=0)


class TestSecurityConfig:
    def test_rate_limit_rpm_positive(self) -> None:
        with pytest.raises(Exception):
            SecurityConfig(rate_limit_rpm=0)

    def test_allowed_origins_parsed_as_list(self) -> None:
        c = SecurityConfig(
            allowed_origins="https://a.com, https://b.com,https://c.com"
        )
        assert c.allowed_origins_list == ["https://a.com", "https://b.com", "https://c.com"]

    def test_allowed_origins_strips_whitespace(self) -> None:
        c = SecurityConfig(allowed_origins="  https://x.com  ,  https://y.com  ")
        assert c.allowed_origins_list == ["https://x.com", "https://y.com"]


# ── Nested config caching ───────────────────────────────────────────────────────

class TestNestedConfig:
    def test_models_property_returns_model_config(self) -> None:
        s = AppSettings()
        assert isinstance(s.models, ModelConfig)

    def test_search_property_returns_search_config(self) -> None:
        s = AppSettings()
        assert isinstance(s.search, SearchConfig)

    def test_reasoning_property_returns_reasoning_config(self) -> None:
        s = AppSettings()
        assert isinstance(s.reasoning, ReasoningConfig)

    def test_security_property_returns_security_config(self) -> None:
        s = AppSettings()
        assert isinstance(s.security, SecurityConfig)

    def test_nested_config_is_cached(self) -> None:
        s = AppSettings()
        c1 = s.models
        c2 = s.models
        assert c1 is c2  # same instance via lru_cache
