"""
Example: web search with the fallback engine.

Run:
    python examples/web_search.py "fusion breakthroughs 2026"

Demonstrates:
- Configuring a custom provider order
- Using the cache to avoid duplicate work
- Inspecting the per-provider metrics
- Closing the engine cleanly

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import sys

from src.search.web_search import FallbackSearchEngine


async def main(query: str, max_results: int) -> None:
    engine = FallbackSearchEngine()
    print(f"\n── Providers in fallback order ──")
    for p in engine._ordered:
        print(f"  - {p.name} (available: {p.is_available})")

    print(f"\n── Searching: {query!r} ──")
    response = await engine.search(query)

    print(f"\nProvider: {response.provider}")
    print(f"Cached:  {response.cached}")
    print(f"Took:    {response.search_time_ms:.0f}ms")
    print(f"Total:   {response.total_results}")
    print()

    for r in response.results[:max_results]:
        print(f"  {r.position:>2}. [{r.provider}] {r.title}")
        print(f"      {r.url}")
        if r.snippet:
            print(f"      {r.snippet[:100]}…")
        print()

    print("── Provider metrics ──")
    for name, m in engine.metrics().items():
        print(f"  {name}: {m}")

    await engine.close()


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "latest fusion energy breakthroughs 2026"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    asyncio.run(main(q, n))
