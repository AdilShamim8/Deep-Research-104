"""
Example: minimal Deep Research pipeline run.

Run:
    python examples/basic_research.py

Requires OPENAI_API_KEY in the environment (or DEEPSEEK_API_KEY with
--provider deepseek). Uses the in-process pipeline; no API server
needed.

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import sys

from src.models import get_model_factory
from src.pipeline.deep_research import DeepResearchPipeline


async def main(question: str, provider: str, model_name: str | None) -> int:
    model = get_model_factory(provider, model_name)
    pipeline = DeepResearchPipeline(model=model, max_iterations=2)
    report = await pipeline.research(question, fast_mode=False)

    print("\n" + "=" * 78)
    print(report.to_markdown())
    print("=" * 78)
    print(f"\nTotal tokens used: {report.total_tokens_used:,}")
    print(f"Total cost (USD):  ${report.total_cost_usd:.6f}")
    print(f"Total LLM calls:   {report.total_llm_calls}")
    print(f"Iterations:        {len(report.iterations)}")
    print(f"Confidence:        {report.confidence}")

    await model.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = "What are the latest breakthroughs in fusion energy as of 2026?"
        provider = "openai"
        model_name = None
    else:
        question = sys.argv[1]
        provider = sys.argv[2] if len(sys.argv) > 2 else "openai"
        model_name = sys.argv[3] if len(sys.argv) > 3 else None

    sys.exit(asyncio.run(main(question, provider, model_name)))
