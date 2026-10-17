"""
Example: streaming research with SSE event handling.

Run:
    python examples/streaming_research.py

Demonstrates how to consume the streaming /research endpoint from
a Python client, parsing Server-Sent Events as they arrive.

Date tag: 17 October 2026.
"""

from __future__ import annotations

import asyncio
import json

import httpx


async def main(question: str, base_url: str = "http://localhost:8000") -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        async with client.stream(
            "POST",
            f"{base_url}/api/v1/research/stream",
            json={"question": question, "provider": "openai", "mode": "deep_research"},
        ) as response:
            print(f"[client] connected | X-Request-ID: "
                  f"{response.headers.get('X-Request-ID', '-')}\n")

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    print("\n[client] stream closed by server")
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "?")
                content = event.get("content", "")
                meta = event.get("metadata", {})
                request_id = event.get("request_id", "-")

                if etype == "complete":
                    print(f"\n[client] FINAL ANSWER:\n{content}\n")
                    print(f"[{request_id}] {meta}")
                else:
                    print(f"[{request_id}] [{etype}] {content}")


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What's new in reasoning models?"
    asyncio.run(main(q))
