"""
Deep Research 104 — autonomous research engine combining web search,
reasoning models (OpenAI o-series, DeepSeek-R1), and inference-time
scaling techniques (CoT, ToT, parallel/sequential sampling, verifier).

Public subpackages:
    src.models     — Unified BaseModel + OpenAI/DeepSeek/Ollama providers
    src.reasoning  — CoT, ToT, parallel/sequential sampling, verifier
    src.search     — Web search + content extraction + aggregation
    src.pipeline   — DeepResearchPipeline + MetaCoT orchestrators
    src.training   — STaR, RL, PRM/ORM, self-refinement
    src.utils      — Token counter, cache, rate limiter
    src.api        — FastAPI app, routes, schemas

Author:  Adil Shamim <adilshamim8@users.noreply.github.com>
Date:    17 October 2026
License: MIT
"""

__version__ = "1.0.0"
__build_date__ = "2026-10-17"
__author__ = "Adil Shamim"
__license__ = "MIT"

__all__ = ["__author__", "__build_date__", "__license__", "__version__"]
