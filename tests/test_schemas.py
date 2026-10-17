"""
Unit tests for src.api.schemas — request validation + response shapes.

Uses pytest's parametrize where useful. No network — purely Pydantic
validation tests.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.api.schemas import (
    CoTRequest,
    CoTResponse,
    ErrorResponse,
    HealthResponse,
    ResearchRequest,
    SearchRequest,
    ToTRequest,
    VersionResponse,
)

# ── ResearchRequest ────────────────────────────────────────────────────────────


class TestResearchRequest:
    def test_valid_request(self) -> None:
        r = ResearchRequest(question="What are fusion breakthroughs?")
        assert r.provider == "openai"
        assert r.max_iterations == 3
        assert r.fast_mode is False
        assert r.mode == "deep_research"

    def test_question_min_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="ab")  # < 5 chars

    def test_question_max_length_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="x" * 2_001)

    def test_question_whitespace_stripped(self) -> None:
        r = ResearchRequest(question="  what is X  ")
        assert r.question == "what is X"

    def test_question_must_not_be_only_whitespace(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="     ")

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="what is X", provider="anthropic")  # type: ignore

    def test_vllm_provider_allowed(self) -> None:
        r = ResearchRequest(question="what is X", provider="vllm")
        assert r.provider == "vllm"

    def test_max_iterations_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="what is X", max_iterations=0)
        with pytest.raises(ValidationError):
            ResearchRequest(question="what is X", max_iterations=11)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResearchRequest(question="what is X", surprise_field="boom")  # type: ignore

    def test_all_modes_allowed(self) -> None:
        for mode in ["deep_research", "meta_cot", "cot", "tot", "self_consistency", "parallel"]:
            r = ResearchRequest(question="what is X", mode=mode)  # type: ignore
            assert r.mode == mode


# ── CoTRequest ──────────────────────────────────────────────────────────────────


class TestCoTRequest:
    def test_valid(self) -> None:
        r = CoTRequest(question="What is 2+2?")
        assert r.cot_mode == "structured"
        assert r.n_samples == 1
        assert r.temperature == 0.7

    def test_n_samples_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CoTRequest(question="q", n_samples=0)
        with pytest.raises(ValidationError):
            CoTRequest(question="q", n_samples=21)

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CoTRequest(question="q", temperature=-0.1)
        with pytest.raises(ValidationError):
            CoTRequest(question="q", temperature=2.5)

    def test_cot_mode_choices(self) -> None:
        with pytest.raises(ValidationError):
            CoTRequest(question="q", cot_mode="invalid")  # type: ignore


# ── ToTRequest ───────────────────────────────────────────────────────────────


class TestToTRequest:
    def test_valid(self) -> None:
        r = ToTRequest(problem="Solve the wolf-goat-cabbage puzzle.")
        assert r.branching_factor == 3
        assert r.max_depth == 4
        assert r.beam_width == 2
        assert r.strategy == "beam"

    def test_branching_factor_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ToTRequest(problem="p", branching_factor=1)
        with pytest.raises(ValidationError):
            ToTRequest(problem="p", branching_factor=7)

    def test_strategy_choices(self) -> None:
        for s in ["beam", "bfs", "dfs"]:
            r = ToTRequest(problem="p", strategy=s)  # type: ignore
            assert r.strategy == s


# ── SearchRequest ────────────────────────────────────────────────────────────


class TestSearchRequest:
    def test_valid(self) -> None:
        r = SearchRequest(query="fusion breakthroughs")
        assert r.max_results == 10
        assert r.fetch_content is False
        assert r.provider is None

    def test_query_min_length(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="a")

    def test_max_results_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(query="q", max_results=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="q", max_results=51)


# ── ErrorResponse ──────────────────────────────────────────────────────────────


class TestErrorResponse:
    def test_from_exception_captures_type(self) -> None:
        err = ErrorResponse.from_exception(ValueError("boom"), request_id="abc")
        assert err.error == "ValueError"
        assert err.detail == "boom"
        assert err.request_id == "abc"
        assert err.timestamp is not None

    def test_from_exception_with_explicit_args(self) -> None:
        err = ErrorResponse.from_exception(
            Exception("x"),
            request_id="r1",
            error_code="custom_code",
            detail="custom detail",
        )
        assert err.error_code == "custom_code"
        assert err.detail == "custom detail"

    def test_defaults(self) -> None:
        err = ErrorResponse(error="x")
        assert err.detail == ""
        assert err.request_id == ""


# ── HealthResponse ────────────────────────────────────────────────────────────


class TestHealthResponse:
    def test_status_must_be_known(self) -> None:
        h = HealthResponse(
            status="healthy",
            python_version="3.11",
            version="1.0.0",
            build_date="2026-10-17",
            timestamp=123.0,
        )
        assert h.status == "healthy"
        with pytest.raises(ValidationError):
            HealthResponse(status="boom", python_version="", version="", build_date="", timestamp=0)  # type: ignore


# ── VersionResponse ────────────────────────────────────────────────────────────


class TestVersionResponse:
    def test_fields(self) -> None:
        v = VersionResponse(
            name="deep-research-104",
            version="1.0.0",
            build_date="2026-10-17",
            python_version="3.11",
            app_env="production",
        )
        assert v.name == "deep-research-104"
        assert v.app_env == "production"


# ── CoTResponse ────────────────────────────────────────────────────────────────


class TestCoTResponse:
    def test_confidence_default(self) -> None:
        r = CoTResponse(answer="4", reasoning="2+2=4")
        assert r.confidence == 1.0
        assert r.supporting_votes == 1
        assert r.total_votes == 1

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CoTResponse(answer="a", reasoning="r", confidence=-0.1)
        with pytest.raises(ValidationError):
            CoTResponse(answer="a", reasoning="r", confidence=1.5)
