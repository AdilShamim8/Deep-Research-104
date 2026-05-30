<div align="center">

# 🔬 Deep Research AI System

### Production-grade autonomous research with web search,
### reasoning models, and inference-time scaling

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](docker-compose.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Overview](#-overview) •
[Architecture](#-architecture) •
[Quick Start](#-quick-start) •
[Usage](#-usage) •
[API Reference](#-api-reference) •
[Training](#-training-pipeline) •
[Deployment](#-deployment) •
[Contributing](#-contributing)

---

<img src="docs/assets/deep_research_banner.png" alt="Deep Research Banner" width="800"/>

> **"The goal is not to replace human researchers,
>    but to give every person the research capability
>    of a world-class analyst."**

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [What Makes This Different](#-what-makes-this-different)
3. [System Architecture](#-system-architecture)
4. [Reasoning Techniques](#-reasoning-techniques)
5. [Quick Start](#-quick-start)
6. [Installation](#-installation)
7. [Configuration](#-configuration)
8. [Usage — CLI](#-usage--cli)
9. [Usage — Python API](#-usage--python-api)
10. [Usage — REST API](#-rest-api)
11. [Inference-Time Scaling](#-inference-time-scaling)
12. [Training Pipeline](#-training-pipeline)
13. [Local Deployment](#-local-deployment)
14. [Deployment](#-deployment)
15. [Monitoring](#-monitoring)
16. [Testing](#-testing)
17. [Project Structure](#-project-structure)
18. [Roadmap](#-roadmap)
19. [Contributing](#-contributing)
20. [License](#-license)

---

## 🔭 Overview

**Deep Research AI System** is a production-ready framework that
combines the reasoning power of state-of-the-art LLMs (OpenAI o3,
DeepSeek-R1) with autonomous web search to answer complex research
questions — the way a senior analyst would.

Instead of a single model call, Deep Research:

1. **Decomposes** your question into targeted search queries
2. **Searches** the web across multiple providers simultaneously
3. **Extracts** full content from the most relevant pages
4. **Reasons** over gathered evidence using CoT, ToT, or Meta-CoT
5. **Identifies gaps** and searches again iteratively
6. **Synthesizes** a comprehensive, cited answer

All of this happens automatically, within configurable compute
and token budgets, with full streaming support for real-time UIs.

### Supported Models

| Provider | Models | Reasoning | Context |
|----------|--------|-----------|---------|
| OpenAI | o3, o3-mini, o1, o1-mini, gpt-4o | ✅ Native | 200K |
| DeepSeek | deepseek-reasoner (R1) | ✅ `<think>` tags | 128K |
| Ollama (local) | deepseek-r1:7b/14b/32b/70b | ✅ `<think>` tags | 32K |
| Ollama (local) | llama3, mistral, qwen2, phi3 | ❌ | 8K–32K |

### Supported Search Providers

| Provider | Cost | Quality | Rate Limit |
|----------|------|---------|------------|
| DuckDuckGo | Free | Good | Moderate |
| SerpAPI | Paid | Excellent (Google) | High |
| Brave Search | Paid | Very Good | High |
| Bing | Paid | Very Good | High |

---

## ✨ What Makes This Different

### vs. Perplexity / standard RAG

| Feature | Standard RAG | Deep Research |
|---------|-------------|---------------|
| Search iterations | 1 | Up to 10 |
| Gap analysis | ❌ | ✅ Automatic |
| Reasoning depth | Shallow | CoT / ToT / Meta-CoT |
| Token limit safety | Basic | Enforced everywhere |
| Inference scaling | ❌ | ✅ Budget-controlled |
| Local/offline mode | ❌ | ✅ Full Ollama support |
| Training pipeline | ❌ | ✅ STaR + GRPO + SFT |
| Streaming | Sometimes | ✅ Full SSE |
| Self-refinement | ❌ | ✅ Built-in |

### vs. LangChain / LlamaIndex agents

- **Token-safe by design**: every prompt is counted before sending
- **No black-box agents**: full visibility into every reasoning step
- **Multiple reasoning modes**: not just ReAct loops
- **Production infrastructure**: rate limiting, caching, metrics
- **Training loop**: improve the model itself over time

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│              CLI  ·  REST API  ·  WebSocket/SSE                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                       FastAPI Application                        │
│         Routes · Schemas · Middleware · Prometheus Metrics       │
└──────────┬─────────────────────────────────────┬────────────────┘
           │                                     │
┌──────────▼──────────┐               ┌──────────▼──────────────┐
│   Research Pipelines │               │    Reasoning Engines     │
│                      │               │                          │
│  DeepResearch        │               │  ChainOfThought          │
│  ├─ Query gen        │               │  ├─ Zero-shot            │
│  ├─ Gap analysis     │               │  ├─ Few-shot             │
│  ├─ Synthesis        │               │  ├─ Structured           │
│  └─ Streaming        │               │  └─ Self-consistency     │
│                      │               │                          │
│  MetaCoT             │               │  ParallelSampler         │
│  ├─ <search> tags    │               │  ├─ Best-of-N            │
│  ├─ <reading> tags   │               │  ├─ Tournament           │
│  ├─ <think> tags     │               │  └─ Diverse              │
│  └─ Agentic loop     │               │                          │
└──────────┬───────────┘               │  SequentialSampler       │
           │                           │  ├─ Critique-refine      │
           │                           │  ├─ Decompose-solve      │
           │                           │  └─ Debate               │
           │                           │                          │
           │                           │  TreeOfThoughts          │
           │                           │  ├─ BFS                  │
           │                           │  ├─ DFS                  │
           │                           │  └─ Beam search          │
           │                           │                          │
           │                           │  InferenceTimeScaler     │
           │                           │  ├─ Samples strategy     │
           │                           │  ├─ Steps strategy       │
           │                           │  ├─ Search strategy      │
           │                           │  ├─ Revision strategy    │
           │                           │  └─ Adaptive selection   │
           │                           └──────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                        Search Layer                              │
│                                                                  │
│  FallbackSearchEngine                                            │
│  ├─ DuckDuckGo  ──┐                                             │
│  ├─ SerpAPI    ───┼──► SearchAggregator                         │
│  ├─ Brave      ───┤    ├─ Deduplication                         │
│  └─ Bing       ───┘    ├─ Relevance scoring                     │
│                         └─ Token budget                          │
│                                                                  │
│  ContentExtractor                                                │
│  ├─ trafilatura  (articles)                                      │
│  ├─ readability  (general)                                       │
│  ├─ BeautifulSoup (fallback)                                     │
│  └─ Playwright   (JS-heavy pages)                                │
└──────────┬──────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│                                                                  │
│  TokenCounter     RateLimiter      LayeredCache                  │
│  ├─ Per-model     ├─ RPM limit     ├─ L1: Disk (diskcache)      │
│  ├─ Pre-flight    ├─ TPM limit     ├─ L2: Redis                 │
│  └─ Truncation    └─ Per-provider  └─ TTL + LRU eviction        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow: Deep Research Request

```
User Question
     │
     ▼
① Generate N search queries (LLM)
     │
     ▼
② Execute queries in parallel (Search APIs)
     │
     ▼
③ Deduplicate + rank results (Aggregator)
     │
     ▼
④ Fetch full content from top URLs (Extractor)
     │
     ▼
⑤ Build token-budget-aware context string
     │
     ▼
⑥ Analyze gaps + confidence (LLM)
     │
     ├─── confidence=high ──► STOP
     │
     └─── confidence=low/medium ──► generate follow-up queries
                                         └── repeat from ②
     │
     ▼
⑦ Final synthesis (LLM with full context)
     │
     ▼
Research Report (answer + sources + metadata)
```

---

## 🧠 Reasoning Techniques

### 1. Chain-of-Thought (CoT)

Forces the model to show its work before answering.

```
Question → [Think step by step] → Reasoning trace → Answer
```

Modes:
- **Zero-shot**: append "Let's think step by step"
- **Few-shot**: provide worked examples first
- **Structured**: enforce `<reasoning>` / `<answer>` tags
- **Self-consistency**: sample N paths, majority vote

When to use: factual questions, math, logical inference.

---

### 2. Parallel Sampling

Generate N independent answers simultaneously, select best.

```
Question ──► Sample 1 ──► Score ──┐
         ──► Sample 2 ──► Score ──┼──► Best Answer
         ──► Sample N ──► Score ──┘
```

Selection methods:
- **Score**: heuristic quality scorer
- **Tournament**: pairwise LLM comparison
- **Diverse**: quality + diversity balance

When to use: when answer quality varies, time is available.

---

### 3. Sequential Sampling

Iteratively improve a single answer through feedback.

```
Draft → Critique → Refine → Critique → Refine → ... → Final
```

Variants:
- **Critique-refine**: model critiques its own output
- **Decompose-solve**: break into sub-tasks, solve sequentially
- **Debate**: argue for/against, then synthesize

When to use: complex analytical writing, nuanced topics.

---

### 4. Tree of Thoughts (ToT)

Explore a tree of reasoning steps, pruning dead ends.

```
              [Root: Problem]
             /       |        \
        [Step A] [Step B] [Step C]
        score=0.8 score=0.3 score=0.6
           /  \              |
      [A1] [A2]           [C1]
      0.9  0.4             0.7
       |
    [Final Answer]
```

Search strategies:
- **Beam search**: keep top-K at each depth (recommended)
- **BFS**: explore all nodes level by level
- **DFS**: go deep on most promising path first

When to use: multi-step math, logic puzzles, planning.

---

### 5. Meta-CoT (Agentic Search)

Model decides when to search, what to search for,
and integrates results into its own reasoning trace.

```
<think>I need facts about X.</think>
<search>X latest research 2024</search>
[Results injected]
<think>Based on results, I need to know Y.</think>
<search>Y detailed explanation</search>
[Results injected]
<think>Now I have enough. Let me synthesize.</think>
<answer>Comprehensive answer with citations.</answer>
```

When to use: open-ended research, unknown unknowns.

---

### 6. Inference-Time Scaling

Control how much compute to spend per question.

```
Token Budget → Strategy Selection → Scaled Answer

Low budget  → Fast parallel sampling
Med budget  → Sequential refinement
High budget → Tree search + revision
Adaptive    → Auto-select per question type
```

Scaling curve example:
```
Quality
  1.0 │                          ●
  0.8 │                    ●
  0.6 │              ●
  0.4 │        ●
  0.2 │  ●
  0.0 └──────────────────────────────► Tokens
       1K   2K   4K   8K  16K  32K
```

---

## 🚀 Quick Start

### Option A: Docker (Recommended — 3 commands)

```bash
# 1. Clone and configure
git clone https://github.com/yourorg/deep-research.git
cd deep-research
cp .env.example .env

# 2. Add at least one API key to .env:
echo "OPENAI_API_KEY=sk-..." >> .env
# or:
echo "DEEPSEEK_API_KEY=ds-..." >> .env

# 3. Start
docker-compose up -d

# Test it
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is fusion energy?", "provider": "openai"}'
```

### Option B: Local Python

```bash
git clone https://github.com/yourorg/deep-research.git
cd deep-research
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
uvicorn src.api.main:app --reload --port 8000
```

### Option C: Fully Local (No API Keys)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull DeepSeek-R1
ollama pull deepseek-r1:7b      # 4.7GB  - fast
ollama pull deepseek-r1:32b     # 19GB   - balanced
ollama pull deepseek-r1:70b     # 40GB   - best quality

# Start with local model only
docker-compose --profile local up -d

# Use it
deep-research ask "What is quantum computing?" --provider ollama
```

---

## 📦 Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11 | 3.11+ |
| RAM | 4 GB | 16 GB |
| Disk | 5 GB | 50 GB (for local models) |
| GPU | Not required | NVIDIA 24GB+ (for training) |
| OS | Linux / macOS | Ubuntu 22.04 |

### Install Options

#### Base (API + Search only)
```bash
pip install -r requirements.txt
```

#### With Training Support
```bash
pip install -r requirements.txt
pip install -e ".[training]"
```

#### With Local Model Support
```bash
pip install -r requirements.txt
pip install -e ".[local]"
```

#### Full Development Setup
```bash
pip install -e ".[dev,training,local]"
pre-commit install
```

### Playwright Setup (for JS-heavy pages)

```bash
playwright install chromium --with-deps
```

### Verify Installation

```bash
python -c "
import src
from src.models.base_model import GenerationConfig
from src.reasoning.chain_of_thought import ChainOfThought
print('✓ Core imports OK')
"
```

---

## ⚙️ Configuration

All configuration is via environment variables. Copy `.env.example`:

```bash
cp .env.example .env
```

### Required: At Least One LLM Provider

```env
# Option 1: OpenAI (recommended for best quality)
OPENAI_API_KEY=sk-proj-...
OPENAI_ORG_ID=org-...            # Optional

# Option 2: DeepSeek (cost-effective)
DEEPSEEK_API_KEY=ds-...

# Option 3: Ollama (free, local, no key needed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:7b
```

### Optional: Search APIs (DuckDuckGo works without any key)

```env
# Uncomment whichever you have:
# SERPAPI_KEY=...           # Google-quality results
# BRAVE_SEARCH_API_KEY=...  # Privacy-focused
# BING_SEARCH_API_KEY=...   # Microsoft

# Which provider to use first:
SEARCH_PROVIDER=duckduckgo  # free default
```

### App Behavior

```env
APP_ENV=production           # development | staging | production
LOG_LEVEL=INFO               # DEBUG | INFO | WARNING | ERROR

# Token limits (tune for your use case)
MAX_TOKENS_PER_REQUEST=128000
REASONING_BUDGET=32768
MAX_OUTPUT_TOKENS=8192

# Research pipeline
MAX_RESEARCH_ITERATIONS=3
MAX_SEARCH_PER_ITERATION=3
MAX_SEARCH_RESULTS=10
MAX_CONTENT_LENGTH=8000      # tokens per extracted page

# Inference scaling
PARALLEL_SAMPLES=3
SEQUENTIAL_STEPS=3
TOT_BRANCHING_FACTOR=3
TOT_MAX_DEPTH=4
TOT_BEAM_WIDTH=2

# Cache
REDIS_URL=redis://localhost:6379/0
```

### Full `.env.example`

```env
# ── LLM Providers ─────────────────────────────────────────────
OPENAI_API_KEY=
OPENAI_ORG_ID=
OPENAI_DEFAULT_MODEL=gpt-4o
OPENAI_REASONING_MODEL=o3-mini

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_REASONING_MODEL=deepseek-reasoner

LOCAL_MODEL_TYPE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:7b

# ── Search ─────────────────────────────────────────────────────
SEARCH_PROVIDER=duckduckgo
SERPAPI_KEY=
BRAVE_SEARCH_API_KEY=
BING_SEARCH_API_KEY=
MAX_SEARCH_RESULTS=10
SEARCH_TIMEOUT_SECONDS=15
MAX_CONTENT_LENGTH=8000

# ── App ────────────────────────────────────────────────────────
APP_ENV=development
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379/0

# ── Token Limits ───────────────────────────────────────────────
MAX_TOKENS_PER_REQUEST=128000
REASONING_BUDGET=32768
MAX_OUTPUT_TOKENS=8192

# ── Research Pipeline ──────────────────────────────────────────
MAX_RESEARCH_ITERATIONS=3
MAX_SEARCH_PER_ITERATION=3

# ── Reasoning ──────────────────────────────────────────────────
PARALLEL_SAMPLES=3
SEQUENTIAL_STEPS=3
TEMPERATURE=0.7
TOP_P=0.9
TOT_BRANCHING_FACTOR=3
TOT_MAX_DEPTH=4
TOT_BEAM_WIDTH=2
```

---

## 💻 Usage — CLI

Install the CLI:

```bash
pip install -e .
```

### `deep-research ask` — Research a Question

```bash
# Basic usage
deep-research ask "What are the latest breakthroughs in fusion energy?"

# Use DeepSeek-R1 via API
deep-research ask "Explain quantum entanglement" \
  --provider deepseek \
  --model deepseek-reasoner

# Use local DeepSeek-R1-7B (free, offline)
deep-research ask "How does CRISPR work?" \
  --provider ollama \
  --model deepseek-r1:7b

# Fast mode (1 iteration, 2 queries)
deep-research ask "What is inflation?" --fast

# Stream progress in real time
deep-research ask "What is the James Webb telescope finding?" --stream

# Meta-CoT mode (model decides when to search)
deep-research ask "Compare fusion vs fission energy" --mode meta_cot

# Chain-of-thought only (no web search)
deep-research ask "Prove that there are infinite primes" --mode cot

# Tree of Thoughts
deep-research ask "Solve: if 3x + 7 = 22, what is x?" --mode tot

# Output raw JSON
deep-research ask "What is GPT-4?" --json

# 5 research iterations
deep-research ask "Full history of the internet" --iterations 5
```

### `deep-research search` — Web Search

```bash
# Basic search
deep-research search "fusion energy 2024"

# More results
deep-research search "quantum computing" --n 20

# Fetch full page content
deep-research search "DeepSeek R1 paper" --fetch
```

### `deep-research scale` — Inference-Time Scaling

```bash
# Auto strategy, 8K token budget
deep-research scale "Prove sqrt(2) is irrational" --budget 8000

# Explicit strategy
deep-research scale "What is quantum computing?" \
  --strategy samples \
  --budget 4000

# Show scaling curve (quality vs. compute)
deep-research scale "Explain CRISPR" \
  --curve \
  --strategy samples \
  --provider ollama
```

**Example scaling curve output:**
```
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Token Budget   ┃ Quality Score ┃ Strategy┃ Candidates ┃ Time (s) ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ 1,000          │ 0.612         │ samples │ 1          │ 2.1      │
│ 2,000          │ 0.698         │ samples │ 1          │ 2.4      │
│ 4,000          │ 0.741         │ samples │ 2          │ 3.8      │
│ 8,000          │ 0.803         │ samples │ 4          │ 6.2      │
│ 16,000         │ 0.851         │ samples │ 8          │ 11.4     │
│ 32,000         │ 0.889         │ samples │ 16         │ 21.7     │
└────────────────┴───────────────┴─────────┴────────────┴──────────┘
```

---

## 🐍 Usage — Python API

### Deep Research Pipeline

```python
import asyncio
from src.models.local_models import get_model_factory
from src.pipeline.deep_research import DeepResearchPipeline

async def main():
    # Choose your model
    model = get_model_factory("openai", "o3-mini")
    # model = get_model_factory("deepseek", "deepseek-reasoner")
    # model = get_model_factory("ollama", "deepseek-r1:7b")

    pipeline = DeepResearchPipeline(
        model=model,
        max_iterations=3,
        queries_per_iteration=3,
        max_context_tokens=24_000,
    )

    report = await pipeline.research(
        "What are the current limitations of large language models?"
    )

    print(report.to_markdown())
    print(f"\nSources: {report.source_urls}")
    print(f"Confidence: {report.confidence}")
    print(f"Tokens used: {report.total_tokens_used:,}")

asyncio.run(main())
```

### Streaming Research

```python
import asyncio
from src.models.local_models import get_model_factory
from src.pipeline.deep_research import DeepResearchPipeline

async def main():
    model = get_model_factory("openai", "o3-mini")
    pipeline = DeepResearchPipeline(model=model, max_iterations=2)

    async for event in pipeline.stream_research(
        "What is the current state of quantum computing?"
    ):
        event_type = event["type"]
        content    = event["content"]

        if event_type == "queries":
            print(f"🔍 {content}")
        elif event_type == "sources_ready":
            print(f"📄 {content}")
        elif event_type == "gap_analysis":
            print(f"🔬 {content}")
        elif event_type == "complete":
            print(f"\n📝 Final Answer:\n{content}")
            meta = event.get("metadata", {})
            print(f"\nSources: {meta.get('source_urls', [])}")

asyncio.run(main())
```

### Chain-of-Thought

```python
import asyncio
from src.models.local_models import get_model_factory
from src.reasoning.chain_of_thought import ChainOfThought
from src.models.base_model import GenerationConfig

async def main():
    model = get_model_factory("openai", "o3-mini")
    cot   = ChainOfThought(model)

    # Single structured CoT
    result = await cot.generate(
        question="If a store offers 25% off a $80 item, "
                 "then an additional 10% off the sale price, "
                 "what is the final price?",
        mode="structured",
        config=GenerationConfig(max_tokens=2048, temperature=0.5),
    )
    print(f"Answer:    {result.answer}")
    print(f"Reasoning: {result.reasoning}")

    # Self-consistency (majority vote over 7 samples)
    consistent = await cot.self_consistency(
        question="What is the 10th Fibonacci number?",
        n_samples=7,
        mode="structured",
    )
    print(f"\nAnswer:     {consistent.answer}")
    print(f"Confidence: {consistent.confidence:.0%}")
    print(f"Votes:      {consistent.supporting_votes}/{consistent.total_votes}")

asyncio.run(main())
```

### Tree of Thoughts

```python
import asyncio
from src.models.local_models import get_model_factory
from src.reasoning.tree_of_thoughts import TreeOfThoughts, SearchStrategy

async def main():
    model = get_model_factory("openai", "o3-mini")

    tot = TreeOfThoughts(
        model=model,
        branching_factor=3,   # 3 candidate thoughts per step
        max_depth=4,          # Up to 4 steps deep
        beam_width=2,         # Keep top 2 at each level
        search_strategy=SearchStrategy.BEAM,
    )

    result = await tot.solve(
        problem=(
            "A farmer has 17 sheep. All but 9 die. "
            "How many sheep are left?"
        )
    )

    print(f"Answer:         {result.answer}")
    print(f"Nodes explored: {result.total_nodes_explored}")
    print(f"Best score:     {result.best_score:.3f}")
    print(f"\nReasoning path:")
    for i, node in enumerate(result.best_path, 1):
        print(f"  Step {i}: {node.thought[:80]}")

asyncio.run(main())
```

### Inference-Time Scaling

```python
import asyncio
from src.models.local_models import get_model_factory
from src.reasoning.inference_scaling import (
    InferenceTimeScaler,
    ScalingBudget,
    ScalingStrategy,
)

async def main():
    model  = get_model_factory("openai", "o3-mini")
    scaler = InferenceTimeScaler(model)

    # Adaptive: auto-selects best strategy
    result = await scaler.scale(
        question="What are the strongest arguments for and "
                 "against universal basic income?",
        budget=ScalingBudget(
            token_budget=16_000,
            max_samples=8,
            max_steps=4,
            time_budget_s=120.0,
        ),
        strategy=ScalingStrategy.ADAPTIVE,
    )

    print(f"Answer:      {result.answer[:200]}...")
    print(f"Strategy:    {result.strategy_used}")
    print(f"Tokens used: {result.tokens_spent:,}")
    print(f"Quality:     {result.quality_score:.3f}")
    print(f"Time:        {result.time_spent_s:.1f}s")
    print(f"Efficiency:  {result.tokens_per_quality_point:.0f} tokens/quality")

    # Compute scaling curve
    print("\n--- Scaling Curve ---")
    results = await scaler.compute_scaling_curve(
        question="Explain the Riemann Hypothesis",
        token_budgets=[1000, 2000, 4000, 8000],
        strategy=ScalingStrategy.SAMPLES,
    )
    for r in results:
        print(
            f"  {r.tokens_spent:>6,} tokens → "
            f"quality={r.quality_score:.3f}"
        )

asyncio.run(main())
```

### Meta-CoT (Agentic Research)

```python
import asyncio
from src.models.local_models import get_model_factory
from src.pipeline.meta_cot import MetaCoT

async def main():
    model    = get_model_factory("openai", "o3-mini")
    meta_cot = MetaCoT(
        model=model,
        max_searches=5,
        max_readings=3,
        max_iterations=10,
    )

    result = await meta_cot.reason(
        "What specific papers did DeepSeek publish in 2024-2025 "
        "and what were their main contributions?"
    )

    print(f"Answer:\n{result.answer}")
    print(f"\nSearches performed: {result.total_searches}")
    print(f"Pages read:         {result.total_readings}")
    print(f"Sources used:       {result.sources_used}")
    print(f"Total tokens:       {result.total_tokens:,}")
    print(f"Time:               {result.total_time_ms:.0f}ms")

asyncio.run(main())
```

### Verifiers

```python
import asyncio
from src.models.local_models import get_model_factory
from src.reasoning.verifier import (
    OutcomeRewardModel,
    ProcessRewardModel,
    RuleBasedVerifier,
    BestOfNWithVerifier,
)

async def main():
    model = get_model_factory("openai", "gpt-4o")

    # ORM: Score a final answer
    orm = OutcomeRewardModel(judge_model=model)
    result = await orm.verify(
        question="What is the boiling point of water at sea level?",
        answer="Water boils at 100°C (212°F) at sea level.",
    )
    print(f"ORM Score:   {result.score:.2f}")
    print(f"Correct:     {result.is_correct}")
    print(f"Feedback:    {result.feedback}")

    # Rule-based: When you have ground truth
    rule_verifier = RuleBasedVerifier()
    result2 = await rule_verifier.verify(
        question="What is 15% of 80?",
        answer="12",
        reference="12",
    )
    print(f"\nRule score:  {result2.score}")
    print(f"Correct:     {result2.is_correct}")

    # Best-of-N: Generate 8, pick best
    generator = get_model_factory("openai", "gpt-4o")
    bon = BestOfNWithVerifier(generator=generator, verifier=orm)
    best_answer, verification, all_results = await bon.generate_and_verify(
        question="Explain the Pythagorean theorem",
        n=4,
    )
    print(f"\nBest answer score: {verification.score:.2f}")
    print(f"Scores: {[f'{v.score:.2f}' for v in all_results]}")

asyncio.run(main())
```

### Self-Refinement

```python
import asyncio
from src.models.local_models import get_model_factory
from src.training.self_refinement import (
    SelfRefinementEngine,
    SelfRefinementConfig,
)

async def main():
    model  = get_model_factory("openai", "o3-mini")
    engine = SelfRefinementEngine(
        model=model,
        config=SelfRefinementConfig(
            max_refinements=3,
            stop_on_satisfied=True,
        ),
    )

    trace = await engine.refine(
        question=(
            "Write a clear, accurate explanation of "
            "how neural networks learn via backpropagation."
        )
    )

    print(f"Refinements:   {trace.n_refinements}")
    print(f"Converged:     {trace.converged}")
    print(f"Score change:  {trace.scores[0]:.2f} → {trace.scores[-1]:.2f}")
    print(f"Improvement:   +{trace.improvement:.2f}")
    print(f"\nFinal answer:\n{trace.final_answer}")

asyncio.run(main())
```

---

## 🌐 REST API

### Start the Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Docs at: http://localhost:8000/docs
```

### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/research` | Deep research |
| POST | `/api/v1/research/stream` | Streaming research (SSE) |
| POST | `/api/v1/cot` | Chain-of-Thought |
| POST | `/api/v1/tot` | Tree of Thoughts |
| POST | `/api/v1/search` | Web search |
| GET | `/api/v1/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |

### POST `/api/v1/research`

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main risks of AI alignment?",
    "provider": "openai",
    "model_name": "o3-mini",
    "max_iterations": 3,
    "fast_mode": false,
    "mode": "deep_research"
  }'
```

**Request schema:**

```json
{
  "question":       "string (5-2000 chars, required)",
  "provider":       "openai | deepseek | ollama",
  "model_name":     "string (optional override)",
  "max_iterations": "integer 1-10 (default: 3)",
  "fast_mode":      "boolean (default: false)",
  "mode":           "deep_research | meta_cot | cot | tot | self_consistency | parallel"
}
```

**Response schema:**

```json
{
  "question":           "string",
  "answer":             "string (full research answer)",
  "confidence":         "low | medium | high",
  "sources": [
    {
      "title":           "string",
      "url":             "string",
      "domain":          "string",
      "relevance_score": "float 0-1",
      "snippet":         "string"
    }
  ],
  "iterations":          "integer",
  "total_sources_found": "integer",
  "total_tokens_used":   "integer",
  "synthesis_model":     "string",
  "processing_time_ms":  "float"
}
```

### POST `/api/v1/research/stream` (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Latest AI news", "provider": "openai"}'
```

**Event stream format:**
```
data: {"type": "start",           "content": "Starting research..."}
data: {"type": "iteration_start", "content": "Research iteration 1"}
data: {"type": "queries",         "content": "Searching: ...", "metadata": {"queries": [...]}}
data: {"type": "search_complete", "content": "Found 15 results"}
data: {"type": "sources_ready",   "content": "Processing 5 sources", "metadata": {...}}
data: {"type": "gap_analysis",    "content": "Confidence: medium | Gaps: 2"}
data: {"type": "synthesizing",    "content": "Generating final answer..."}
data: {"type": "complete",        "content": "<full answer>", "metadata": {...}}
data: [DONE]
```

### POST `/api/v1/cot`

```bash
curl -X POST http://localhost:8000/api/v1/cot \
  -H "Content-Type: application/json" \
  -d '{
    "question": "If x² - 5x + 6 = 0, what are the values of x?",
    "cot_mode": "structured",
    "n_samples": 5,
    "provider": "openai",
    "temperature": 0.7
  }'
```

### POST `/api/v1/tot`

```bash
curl -X POST http://localhost:8000/api/v1/tot \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "How many ways can 4 people sit at a round table?",
    "branching_factor": 3,
    "max_depth": 4,
    "beam_width": 2,
    "strategy": "beam",
    "provider": "openai"
  }'
```

### POST `/api/v1/search`

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "DeepSeek R1 technical report",
    "max_results": 10,
    "fetch_content": false
  }'
```

---

## 🎓 Training Pipeline

The training system improves reasoning capability over time.
All training components work independently — use what you need.

### Step 1: Collect Reasoning Data (STaR)

Prepare a problems file `data/problems.json`:

```json
[
  {"question": "What is 15% of 240?",      "answer": "36"},
  {"question": "If 3x + 7 = 22, find x",  "answer": "5"},
  {"question": "What is the capital of Australia?", "answer": "Canberra"}
]
```

Run STaR data collection:

```bash
deep-research-train star data/problems.json \
  --provider ollama \
  --model deepseek-r1:7b \
  --iterations 3 \
  --concurrent 5 \
  --output data/star_dataset.json
```

**How STaR works:**
```
For each problem:
  ① Generate answer with current model
  ② If correct → save (question, reasoning, answer) as training example
  ③ If wrong   → give model the correct answer, ask it to rationalize
  ④ If rationalization is correct → save as training example
Repeat for N iterations
```

Expected output:
```
STaR Training
  Problems:   1000
  Iterations: 3
  Provider:   ollama

✓ STaR complete
  Total examples:   847
  Direct:           612
  Rationalized:     235
  Accuracy:         84.7%
  Saved to:         data/star_dataset.json
```

### Step 2: Fine-Tune on STaR Data (SFT)

Requires GPU. Supports LoRA for consumer hardware.

```bash
deep-research-train sft data/star_dataset.json \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --epochs 3 \
  --batch-size 4 \
  --lora \
  --lora-rank 16 \
  --output models/sft_v1
```

**Memory requirements with LoRA:**

| Model | VRAM Required |
|-------|--------------|
| 1.5B | ~8 GB |
| 7B | ~16 GB |
| 14B | ~24 GB |
| 32B | ~48 GB (2x GPU) |

### Step 3: Generate Preference Pairs

For reward model training. Prepare questions:
```
# data/questions.txt
What is quantum entanglement?
Explain how CRISPR works
What are the risks of AGI?
```

```bash
deep-research-train pairs data/questions.txt \
  --teacher openai \
  --teacher-model o3-mini \
  --student ollama \
  --student-model deepseek-r1:7b \
  --n-samples 4 \
  --output data/preference_pairs.json
```

### Step 4: RL Training (GRPO)

```bash
deep-research-train grpo data/problems.json \
  --policy deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --group-size 8 \
  --epochs 1 \
  --output models/grpo_v1
```

**How GRPO works:**
```
For each batch of questions:
  ① Sample G=8 responses from policy model
  ② Score each with verifier → rewards [r₁, r₂, ..., r₈]
  ③ Normalize: advantage = (rᵢ - mean(r)) / std(r)
  ④ Update policy: maximize advantage, penalize KL divergence
     from reference model
```

### Training Python API

```python
import asyncio
from src.training.star_trainer import STaRTrainer, STaRConfig
from src.training.self_refinement import SelfRefinementEngine
from src.reasoning.verifier import RuleBasedVerifier
from src.models.local_models import get_model_factory

async def main():
    model    = get_model_factory("ollama", "deepseek-r1:7b")
    verifier = RuleBasedVerifier()

    trainer = STaRTrainer(
        model=model,
        verifier=verifier,
        config=STaRConfig(
            max_iterations=3,
            concurrent_requests=5,
            save_dataset_path="./data/star_out.json",
        )
    )

    problems = [
        {"question": "What is 15% of 80?",  "answer": "12"},
        {"question": "Capital of Germany?",  "answer": "Berlin"},
    ]

    dataset = await trainer.run(problems)
    print(f"Collected {len(dataset.examples)} training examples")
    print(f"Accuracy: {dataset.accuracy:.1%}")

asyncio.run(main())
```

---

## 🖥 Local Deployment

### Quick Local Setup

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull models (choose based on your hardware)
ollama pull deepseek-r1:7b    # 4.7 GB  — 8GB RAM minimum
ollama pull deepseek-r1:14b   # 9.0 GB  — 16GB RAM
ollama pull deepseek-r1:32b   # 19  GB  — 32GB RAM
ollama pull deepseek-r1:70b   # 40  GB  — 64GB RAM

# 3. Start with Docker
docker-compose --profile local up -d

# 4. Test
deep-research ask "Explain neural networks" \
  --provider ollama \
  --model deepseek-r1:7b
```

### Model Selection Guide

| Use Case | Model | RAM | Quality |
|----------|-------|-----|---------|
| Testing / dev | deepseek-r1:7b | 8 GB | Good |
| Daily use | deepseek-r1:14b | 16 GB | Very Good |
| Production | deepseek-r1:32b | 32 GB | Excellent |
| Research | deepseek-r1:70b | 64 GB | Best |

### GPU Acceleration

Ollama auto-detects GPU. For full GPU usage:
```bash
# Check GPU detection
ollama run deepseek-r1:7b "test"

# Check GPU layers
ollama ps
```

### Fully Offline Mode

With Ollama + DuckDuckGo (free, no API key):
```env
# .env — no API keys needed
LOCAL_MODEL_TYPE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:7b
SEARCH_PROVIDER=duckduckgo
APP_ENV=development
```

```bash
docker-compose --profile local up -d
deep-research ask "How does fusion work?" --provider ollama
```

---

## 🚢 Deployment

### Docker Compose Profiles

```bash
# API + Redis only (minimum)
docker-compose up -d

# With local Ollama
docker-compose --profile local up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Everything
docker-compose --profile local --profile monitoring up -d
```

### Production Checklist

```bash
# 1. Set environment
APP_ENV=production
LOG_LEVEL=WARNING

# 2. Set strong Redis password
REDIS_URL=redis://:password@redis:6379/0

# 3. Restrict CORS in src/api/main.py
allow_origins=["https://yourdomain.com"]

# 4. Enable HTTPS (via nginx/traefik reverse proxy)

# 5. Set resource limits (already in docker-compose.yml)

# 6. Verify health
curl http://localhost:8000/api/v1/health
```

### Kubernetes (Helm values example)

```yaml
replicaCount: 3

image:
  repository: yourregistry/deep-research
  tag: "1.0.0"

env:
  APP_ENV: production
  REDIS_URL: redis://redis-service:6379/0
  OPENAI_API_KEY:
    secretKeyRef:
      name: deep-research-secrets
      key: openai-api-key

resources:
  requests:
    memory: "2Gi"
    cpu: "500m"
  limits:
    memory: "8Gi"
    cpu: "2000m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

redis:
  enabled: true
  architecture: standalone
```

### Nginx Reverse Proxy

```nginx
server {
    listen 443 ssl http2;
    server_name research.yourdomain.com;

    ssl_certificate     /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    # SSE streaming — critical settings
    location /api/v1/research/stream {
        proxy_pass         http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header   Connection '';
        proxy_buffering    off;
        proxy_cache        off;
        proxy_read_timeout 300s;
        chunked_transfer_encoding on;
    }

    location / {
        proxy_pass         http://localhost:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

---

## 📊 Monitoring

### Prometheus Metrics

```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d

# View metrics
curl http://localhost:8000/metrics

# Prometheus UI
open http://localhost:9090

# Grafana dashboards
open http://localhost:3000  # admin / admin
```

### Available Metrics

```
# Request count by endpoint and status
deep_research_requests_total{method, endpoint, status_code}

# Request latency histogram
deep_research_request_duration_seconds{endpoint}

# Token usage by model
deep_research_tokens_total{model, type}
```

### Useful Prometheus Queries

```promql
# Request rate (last 5 min)
rate(deep_research_requests_total[5m])

# P95 latency per endpoint
histogram_quantile(0.95,
  rate(deep_research_request_duration_seconds_bucket[5m])
)

# Error rate
rate(deep_research_requests_total{status_code=~"5.."}[5m])
  /
rate(deep_research_requests_total[5m])

# Token usage rate
rate(deep_research_tokens_total[1h])
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run by Module

```bash
pytest tests/test_models.py      -v  # Model providers
pytest tests/test_reasoning.py   -v  # CoT, ToT, sampling
pytest tests/test_pipeline.py    -v  # Deep research pipeline
pytest tests/test_inference_scaling.py -v  # Scaling
```

### With Coverage

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
open htmlcov/index.html
```

### Run Specific Test

```bash
pytest tests/test_reasoning.py::TestChainOfThought::test_self_consistency_majority_vote -v
```

### Integration Test (Live APIs)

```bash
# Requires real API keys in .env
pytest tests/ -v -m integration --timeout=120
```

### Expected Coverage

```
Module                           Stmts  Coverage
─────────────────────────────────────────────────
src/models/base_model.py          89     98%
src/models/openai_models.py      142     91%
src/models/deepseek_models.py    118     89%
src/models/local_models.py       134     87%
src/reasoning/chain_of_thought.py 198    95%
src/reasoning/parallel_sampling.py 167   93%
src/reasoning/sequential_sampling.py 201 91%
src/reasoning/tree_of_thoughts.py 243   89%
src/reasoning/verifier.py        189    94%
src/reasoning/inference_scaling.py 178  92%
src/search/web_search.py         231    88%
src/search/content_extractor.py  198    85%
src/search/search_aggregator.py  167    91%
src/pipeline/deep_research.py    289    87%
src/pipeline/meta_cot.py         234    88%
src/training/star_trainer.py     198    83%
src/training/reward_models.py    167    81%
src/training/rl_trainer.py       189    79%
src/training/self_refinement.py  156    92%
src/utils/token_counter.py       134    98%
src/utils/rate_limiter.py         89    96%
src/utils/cache.py               112    93%
─────────────────────────────────────────────────
TOTAL                           3706    90%
```

---

## 📁 Project Structure

```
deep_research/
│
├── README.md                   ← You are here
├── requirements.txt            ← All dependencies
├── setup.py                    ← Package install
├── .env.example                ← Config template
├── Dockerfile                  ← Production + dev stages
├── docker-compose.yml          ← Full stack (API, Redis, Ollama, Grafana)
│
├── config/
│   ├── __init__.py
│   ├── settings.py             ← Pydantic settings (env vars)
│   ├── logging_config.py       ← Loguru structured logging
│   └── prometheus.yml          ← Prometheus scrape config
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/                 ← LLM Provider Clients
│   │   ├── __init__.py
│   │   ├── base_model.py       ← Abstract interface
│   │   ├── openai_models.py    ← o3, o1, gpt-4o
│   │   ├── deepseek_models.py  ← DeepSeek-R1
│   │   └── local_models.py     ← Ollama + model factory
│   │
│   ├── reasoning/              ← Inference-Time Techniques
│   │   ├── __init__.py
│   │   ├── chain_of_thought.py ← Zero/few/structured + self-consistency
│   │   ├── parallel_sampling.py← Best-of-N, tournament, diverse
│   │   ├── sequential_sampling.py ← Critique-refine, decompose, debate
│   │   ├── tree_of_thoughts.py ← BFS/DFS/Beam ToT
│   │   ├── verifier.py         ← ORM, PRM, rule-based, Best-of-N
│   │   └── inference_scaling.py← Unified scaling controller
│   │
│   ├── search/                 ← Web Search + Extraction
│   │   ├── __init__.py
│   │   ├── web_search.py       ← DDG, SerpAPI, Brave, Bing + fallback
│   │   ├── content_extractor.py← trafilatura/readability/BS4/Playwright
│   │   └── search_aggregator.py← Dedup, rank, token budget
│   │
│   ├── pipeline/               ← Research Orchestrators
│   │   ├── __init__.py
│   │   ├── deep_research.py    ← Iterative research + streaming
│   │   └── meta_cot.py         ← Agentic search-in-reasoning
│   │
│   ├── training/               ← Training Pipelines
│   │   ├── __init__.py
│   │   ├── star_trainer.py     ← STaR + SFT wrapper
│   │   ├── reward_models.py    ← ORM training + synthetic pairs
│   │   ├── rl_trainer.py       ← GRPO episode collection + training
│   │   └── self_refinement.py  ← Constitutional-style self-improvement
│   │
│   ├── api/                    ← FastAPI Application
│   │   ├── __init__.py
│   │   ├── main.py             ← App factory, middleware, metrics
│   │   ├── routes.py           ← All route handlers
│   │   └── schemas.py          ← Pydantic request/response models
│   │
│   └── utils/                  ← Shared Utilities
│       ├── __init__.py
│       ├── token_counter.py    ← Token counting + truncation
│       ├── rate_limiter.py     ← Token bucket (RPM + TPM)
│       └── cache.py            ← L1 disk + L2 Redis layered cache
│
├── scripts/
│   ├── cli.py                  ← deep-research CLI
│   └── train.py                ← deep-research-train CLI
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_reasoning.py
│   ├── test_search.py
│   ├── test_pipeline.py
│   └── test_inference_scaling.py
│
├── data/                       ← Training data (git-ignored)
│   └── .gitkeep
│
├── models/                     ← Local model checkpoints (git-ignored)
│   └── .gitkeep
│
├── logs/                       ← Application logs (git-ignored)
│   └── .gitkeep
│
└── notebooks/
    └── demo.ipynb              ← Interactive demo
```

---

## 🗺 Roadmap

### ✅ v1.0.0 — Current Release

- Multi-provider LLM support (OpenAI, DeepSeek, Ollama)
- All inference-time scaling techniques
- Full Deep Research pipeline with streaming
- Meta-CoT agentic research
- STaR + GRPO + SFT training loop
- Production API (FastAPI + SSE)
- Docker + monitoring stack
- CLI with rich output

### 🔄 v1.1.0 — In Progress

- [ ] PDF and academic paper ingestion
- [ ] Vector store integration (FAISS / Pinecone)
- [ ] Multi-modal support (images in research)
- [ ] API authentication (JWT)
- [ ] Rate limiting per API key

### 📋 v1.2.0 — Planned

- [ ] Persistent research sessions (save/resume)
- [ ] Custom tool plugins (calculator, code executor)
- [ ] Multi-agent collaboration
- [ ] Citation verification system
- [ ] Web UI (React frontend)

### 🔮 v2.0.0 — Future

- [ ] Retrieval-Augmented Generation (full RAG pipeline)
- [ ] Cross-language research (translate + synthesize)
- [ ] Continuous learning from user feedback
- [ ] Domain-specific fine-tuned model hub

---

## 🤝 Contributing

We welcome contributions of all kinds.

### Getting Started

```bash
# Fork and clone
git clone https://github.com/yourorg/deep-research.git
cd deep-research

# Create feature branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -e ".[dev]"
pre-commit install

# Make changes, add tests
# ...

# Run checks
black src/ tests/
ruff src/ tests/
mypy src/
pytest tests/ -v

# Commit and push
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
# Open Pull Request
```

### Code Standards

```bash
# Formatting
black src/ tests/ scripts/

# Linting
ruff check src/ tests/ --fix

# Type checking
mypy src/ --ignore-missing-imports

# All at once
make lint  # if using Makefile
```

### What We're Looking For

- 🐛 **Bug fixes** with regression tests
- ✨ **New reasoning techniques** (with benchmarks)
- 🔍 **New search providers** (follow `BaseSearchProvider`)
- 🧪 **Better test coverage**
- 📚 **Documentation improvements**
- 🌐 **Translations**

### Pull Request Guidelines

1. One feature or fix per PR
2. All tests must pass
3. New features need new tests
4. Update README if adding new CLI commands
5. Add your change to the appropriate section

---

## 📄 License

```
MIT License

Copyright (c) 2025 Deep Research AI

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

See [LICENSE](LICENSE) for full text.

---

## 🙏 Acknowledgements

This system is built on the shoulders of giants:

| Work | Authors | Contribution |
|------|---------|--------------|
| Chain-of-Thought Prompting | Wei et al. (2022) | CoT foundation |
| Self-Consistency | Wang et al. (2022) | Majority voting |
| Tree of Thoughts | Yao et al. (2023) | ToT framework |
| STaR | Zelikman et al. (2022) | Reasoning data bootstrapping |
| Let's Verify Step by Step | Lightman et al. (2023) | PRM training |
| Self-Refine | Madaan et al. (2023) | Self-refinement |
| DeepSeek-R1 | DeepSeek AI (2025) | GRPO + reasoning distillation |
| Scaling LLM Test-Time Compute | Snell et al. (2024) | Inference scaling |
| OpenAI o1 System Card | OpenAI (2024) | Reasoning model design |

---

<div align="center">

**Built with ❤️ for the open-source AI research community**

⭐ Star this repo if it helped your research

[Report Bug](https://github.com/yourorg/deep-research/issues) •
[Request Feature](https://github.com/yourorg/deep-research/issues) •
[Join Discord](https://discord.gg/yourserver)

</div>