"""
Unit tests for src.models.base_model — BaseModel contract, cost tracking,
capability assertion, GenerationConfig validation, ModelResponse helpers.
"""

from __future__ import annotations

import pytest

from src.models.base_model import (
    BaseModel,
    GenerationConfig,
    ModelCapability,
    ModelResponse,
    PRICING_TABLE,
    estimate_cost,
)


# ── GenerationConfig ──────────────────────────────────────────────────────────

class TestGenerationConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = GenerationConfig()
        assert cfg.max_tokens == 4_096
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.stream is False
        assert cfg.n == 1

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(max_tokens=0)
        with pytest.raises(ValueError):
            GenerationConfig(max_tokens=-1)

    def test_temperature_bounds(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            GenerationConfig(temperature=2.5)
        # Edges are valid
        GenerationConfig(temperature=0.0)
        GenerationConfig(temperature=2.0)

    def test_top_p_bounds(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(top_p=0.0)
        with pytest.raises(ValueError):
            GenerationConfig(top_p=1.5)
        # 1.0 is valid (inclusive upper bound)
        GenerationConfig(top_p=1.0)

    def test_n_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            GenerationConfig(n=0)

    def test_reasoning_effort_default_none(self) -> None:
        cfg = GenerationConfig()
        assert cfg.reasoning_effort is None
        assert cfg.budget_tokens is None

    def test_json_schema_default_none(self) -> None:
        cfg = GenerationConfig()
        assert cfg.json_schema is None
        assert cfg.user is None


# ── ModelResponse ──────────────────────────────────────────────────────────────

class TestModelResponse:
    def test_defaults(self) -> None:
        r = ModelResponse(content="hi")
        assert r.content == "hi"
        assert r.reasoning_content is None
        assert r.model == ""
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.total_tokens == 0
        assert r.alternatives == []
        assert r.cached is False
        assert r.cost_usd == 0.0
        assert r.total_cost_estimate == r.cost_usd

    def test_to_log_dict_excludes_content(self) -> None:
        r = ModelResponse(content="secret answer", model="gpt-4o", prompt_tokens=10)
        d = r.to_log_dict()
        assert "content" not in d
        assert d["model"] == "gpt-4o"
        assert d["prompt_tokens"] == 10


# ── Capability registry ────────────────────────────────────────────────────────

class TestModelCapability:
    def test_all_capabilities_have_unique_values(self) -> None:
        values = [cap.value for cap in ModelCapability]
        assert len(values) == len(set(values))

    def test_capabilities_set_is_immutable_lookup(self) -> None:
        class FakeModel(BaseModel):
            async def generate(self, messages, config=None):
                return ModelResponse(content="x")

            async def stream_generate(self, messages, config=None):
                yield "x"

            def count_tokens(self, text):
                return 1

            def get_context_limit(self):
                return 8192

        m = FakeModel("test")
        m.capabilities = {ModelCapability.STREAMING, ModelCapability.REASONING}
        assert m.has_capability(ModelCapability.STREAMING)
        assert m.has_capability(ModelCapability.REASONING)
        assert not m.has_capability(ModelCapability.VISION)

    def test_require_capability_raises(self) -> None:
        class FakeModel(BaseModel):
            async def generate(self, messages, config=None):
                return ModelResponse(content="x")

            async def stream_generate(self, messages, config=None):
                yield "x"

            def count_tokens(self, text):
                return 1

            def get_context_limit(self):
                return 8192

        m = FakeModel("test")
        m.capabilities = {ModelCapability.STREAMING}
        with pytest.raises(NotImplementedError):
            m.require_capability(ModelCapability.VISION)
        # Should not raise
        m.require_capability(ModelCapability.STREAMING)


# ── Cost estimation ──────────────────────────────────────────────────────────

class TestCostEstimation:
    def test_known_model_returns_positive_cost(self) -> None:
        # o3-mini: $0.0011 / 1K input + $0.0044 / 1K output
        cost = estimate_cost("o3-mini", prompt_tokens=1000, completion_tokens=1000)
        assert cost > 0
        assert cost == pytest.approx(0.0011 + 0.0044, rel=0.01)

    def test_deepseek_reasoner_cheaper_than_o3(self) -> None:
        o3 = estimate_cost("o3", prompt_tokens=10_000, completion_tokens=10_000)
        ds = estimate_cost(
            "deepseek-reasoner", prompt_tokens=10_000, completion_tokens=10_000
        )
        assert ds < o3

    def test_unknown_model_returns_zero(self) -> None:
        assert estimate_cost("totally-unknown-xyz", 1000, 1000) == 0.0

    def test_prefix_match_used(self) -> None:
        # "o3-mini-2025-01-31" should match "o3-mini"
        cost_with_suffix = estimate_cost(
            "o3-mini-2025-01-31", prompt_tokens=1000, completion_tokens=1000
        )
        cost_canonical = estimate_cost(
            "o3-mini", prompt_tokens=1000, completion_tokens=1000
        )
        assert cost_with_suffix == cost_canonical

    def test_zero_tokens_zero_cost(self) -> None:
        assert estimate_cost("o3-mini", 0, 0) == 0.0

    def test_pricing_table_has_all_known_models(self) -> None:
        for model in [
            "o3-mini", "o3", "o1", "deepseek-reasoner",
            "deepseek-chat", "gpt-4o", "gpt-4o-mini",
        ]:
            assert model in PRICING_TABLE


# ── update_cost integration ────────────────────────────────────────────────────

class TestUpdateCost:
    def test_update_cost_populates_response(self) -> None:
        class FakeModel(BaseModel):
            async def generate(self, messages, config=None):
                return ModelResponse(content="x")

            async def stream_generate(self, messages, config=None):
                yield "x"

            def count_tokens(self, text):
                return 1

            def get_context_limit(self):
                return 8192

        m = FakeModel("o3-mini")
        response = ModelResponse(
            content="answer", model="o3-mini",
            prompt_tokens=1000, completion_tokens=1000,
        )
        m.update_cost(response)
        assert response.cost_usd > 0
        assert response.cost_usd == pytest.approx(0.0011 + 0.0044, rel=0.01)
