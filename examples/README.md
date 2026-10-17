# Examples

Runnable scripts that demonstrate how to use Deep Research 104
programmatically. Each script is self-contained and prints informative
output to stdout.

> **Build date:** 17 October 2026

## Scripts

| Script | What it shows | API key needed |
|--------|---------------|----------------|
| `basic_research.py` | Minimal end-to-end pipeline run with Markdown + cost report | Yes (OPENAI or DEEPSEEK) |
| `streaming_research.py` | SSE client that consumes `/research/stream` events | Yes (running API server) |
| `cot_reasoning.py` | Single-shot CoT + self-consistency voting | Yes |
| `web_search.py` | Fallback search engine + per-provider metrics inspection | No (uses free DuckDuckGo) |

## Running

```bash
# Install the package in editable mode
pip install -e .

# Set your API key
export OPENAI_API_KEY=sk-...

# Run an example
python examples/basic_research.py "What are the latest fusion breakthroughs?"
python examples/cot_reasoning.py "If a train travels 120 miles in 2 hours, what's its average speed?"
python examples/web_search.py "fusion breakthroughs 2026"

# Streaming example requires the API server running
uvicorn src.api.main:app --port 8000 &
python examples/streaming_research.py "What's new in reasoning models?"
```

## Notes

- All examples honor `APP_ENV=development` defaults so they won't
  refuse to start because of missing production-only env vars.
- The `web_search.py` example is the only one that doesn't need an
  API key — it uses DuckDuckGo (free, no key).
- Streaming example uses `httpx.AsyncClient` with a 10-minute timeout
  because research runs can take several minutes.
- Each example closes its resources cleanly (model.close(),
  engine.close()) so it can be looped without leaking connections.
