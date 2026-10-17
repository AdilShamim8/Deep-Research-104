"""
Example: Chain-of-Thought + self-consistency reasoning.

Run:
    python examples/cot_reasoning.py "What is the sum of all primes below 100?"

Demonstrates zero-shot CoT, structured CoT, and self-consistency
voting over 5 samples. Useful for understanding the inference-time
scaling knobs.

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import sys

from src.models import get_model_factory
from src.models.base_model import GenerationConfig
from src.reasoning.chain_of_thought import ChainOfThought


async def main(question: str, provider: str, n_samples: int) -> None:
    model = get_model_factory(provider)
    engine = ChainOfThought(model)

    config = GenerationConfig(max_tokens=4_096, temperature=0.8)

    print("\n── Single-shot structured CoT ──")
    result = await engine.generate(question=question, mode="structured", config=config)
    print(f"\nReasoning:\n{result.reasoning}\n")
    print(f"Answer: {result.answer}\n")

    if n_samples > 1:
        print(f"\n── Self-consistency over {n_samples} samples ──")
        sc = await engine.self_consistency(
            question=question, n_samples=n_samples, mode="structured", config=config,
        )
        print(f"\nReasoning (best):\n{sc.reasoning}\n")
        print(f"Answer: {sc.answer}")
        print(f"Confidence: {sc.confidence:.2f} ({sc.supporting_votes}/{sc.total_votes} votes)")

    await model.close()


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "If a train travels 120 miles in 2 hours, what's its average speed?"
    p = sys.argv[2] if len(sys.argv) > 2 else "openai"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    asyncio.run(main(q, p, n))
