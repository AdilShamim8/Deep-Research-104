"""
Test package for Deep Research 104.

Test layout:
    tests/test_config.py         — AppSettings + nested config validation
    tests/test_schemas.py        — Pydantic request/response schemas
    tests/test_search.py         — Search layer (web_search + aggregator)
    tests/test_utils.py          — LayeredCache + LRUCache + key derivation
    tests/test_token_counter.py  — TokenCounter + truncation
    tests/test_models.py         — BaseModel + provider clients (mocked)
    tests/test_reasoning.py      — CoT + ToT + sampling + verifier
    tests/test_pipeline.py       — DeepResearchPipeline + MetaCoT (mocked)

Marks:
    @pytest.mark.unit            — fast, no network (default in CI)
    @pytest.mark.integration     — requires Redis, OpenAI, or Ollama
    @pytest.mark.slow             — >5s

Run: pytest tests/ -m "not slow and not integration"
"""
