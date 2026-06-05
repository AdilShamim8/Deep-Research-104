<div align="center">

# Project 4: Build "Deep Research" Capability with Web Search and Reasoning Models

**Production-grade autonomous research engine combining**
**web search, reasoning models, and inference-time scaling**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](docker-compose.yml)
[![DeepSeek-R1](https://img.shields.io/badge/DeepSeek--R1-supported-purple.svg)](https://github.com/deepseek-ai/DeepSeek-R1)
[![OpenAI o3](https://img.shields.io/badge/OpenAI-o3%20%7C%20o1-412991.svg)](https://openai.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linting-ruff-FCC21B.svg)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](htmlcov/index.html)

<br/>

[**Overview**](#-overview) •
[**Architecture**](#-system-architecture) •
[**Quick Start**](#-quick-start) •
[**Installation**](#-installation) •
[**Configuration**](#-configuration) •
[**CLI**](#-cli-reference) •
[**Python API**](#-python-api) •
[**REST API**](#-rest-api) •
[**Reasoning**](#-reasoning-techniques) •
[**Scaling**](#-inference-time-scaling) •
[**Training**](#-training-pipeline) •
[**Local**](#-local-deployment) •
[**Deploy**](#-production-deployment) •
[**Testing**](#-testing) •
[**Structure**](#-project-structure) •
[**Roadmap**](#-roadmap)

<br/>

> *"The goal is not to replace human researchers —*
> *it is to give every person the research capability*
> *of a world-class analyst."*

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [What Makes This Different](#-what-makes-this-different)
3. [System Architecture](#-system-architecture)
4. [Reasoning Techniques](#-reasoning-techniques)
5. [Inference-Time Scaling](#-inference-time-scaling)
6. [Quick Start](#-quick-start)
7. [Installation](#-installation)
8. [Configuration](#-configuration)
9. [CLI Reference](#-cli-reference)
10. [Python API](#-python-api)
11. [REST API](#-rest-api)
12. [Training Pipeline](#-training-pipeline)
13. [Local Deployment](#-local-deployment)
14. [Production Deployment](#-production-deployment)
15. [Monitoring](#-monitoring)
16. [Testing](#-testing)
17. [Project Structure](#-project-structure)
18. [Roadmap](#-roadmap)
19. [Contributing](#-contributing)
20. [Acknowledgements](#-acknowledgements)
21. [License](#-license)

---

## 🔭 Overview

**Deep Research AI System** is a production-ready framework that
combines the reasoning power of state-of-the-art language models
— OpenAI o3, DeepSeek-R1, and local Ollama models — with
autonomous multi-iteration web search to answer complex research
questions the way a senior analyst would.

Instead of a single model call, Deep Research orchestrates a
full research workflow:

```
Your Question
      │
      ▼
① Decompose into targeted search queries
      │
      ▼
② Search web in parallel (4 providers with fallback)
      │
      ▼
③ Extract full content from top URLs
      │
      ▼
④ Reason over evidence (CoT / ToT / Meta-CoT)
      │
      ▼
⑤ Identify gaps → search again (up to 10 iterations)
      │
      ▼
⑥ Synthesize comprehensive, cited final answer
```

Everything runs within configurable token budgets, with full
streaming support for real-time UIs and CLI progress display.

---

### Supported Models

| Provider | Models | Reasoning | Context Window |
|----------|--------|:---------:|:--------------:|
| **OpenAI** | o3, o3-mini, o1, o1-mini, gpt-4o | ✅ Native | 200K |
| **DeepSeek** | deepseek-reasoner (R1) | ✅ `<think>` tags | 128K |
| **Ollama** | deepseek-r1:7b / 14b / 32b / 70b | ✅ `<think>` tags | 32K |
| **Ollama** | llama3, mistral, qwen2, phi3 | — | 8K–32K |

### Supported Search Providers

| Provider | Cost | Quality | Notes |
|----------|:----:|:-------:|-------|
| **DuckDuckGo** | Free | Good | Default — no key needed |
| **SerpAPI** | Paid | Excellent | Google results |
| **Brave Search** | Paid | Very Good | Privacy-focused |
| **Bing** | Paid | Very Good | Microsoft index |

Providers fall back automatically if one fails or returns no results.

---

## ✨ What Makes This Different

### vs. Standard RAG / Perplexity

| Feature | Standard RAG | **Deep Research** |
|---------|:-----------:|:-----------------:|
| Search iterations | 1 | Up to 10 |
| Gap analysis | ❌ | ✅ Automatic |
| Reasoning depth | Shallow | CoT / ToT / Meta-CoT |
| Token limit safety | Basic | Enforced at every call |
| Inference-time scaling | ❌ | ✅ Budget-controlled |
| Local / offline mode | ❌ | ✅ Full Ollama support |
| Training pipeline | ❌ | ✅ STaR + GRPO + SFT |
| Streaming | Sometimes | ✅ Full SSE |
| Self-refinement | ❌ | ✅ Constitutional-style |
| Verifier (ORM + PRM) | ❌ | ✅ Built-in |

### vs. LangChain / LlamaIndex Agents

- **Token-safe by design** — every prompt counted before sending
- **No black-box agents** — full visibility into every step
- **Multiple reasoning modes** — not just ReAct loops
- **Production infrastructure** — rate limiting, caching, metrics
- **Training loop** — improve the model itself over time

---

## 🏗 System Architecture

### Component Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            CLIENT LAYER                                  │
│                  CLI  ·  REST API  ·  SSE Streaming                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                         FASTAPI APPLICATION                              │
│         Routes · Pydantic Schemas · Middleware · Prometheus Metrics      │
└──────────────────┬─────────────────────────────────┬────────────────────┘
                   │                                 │
   ┌───────────────▼───────────────┐   ┌─────────────▼──────────────────┐
   │       RESEARCH PIPELINES      │   │        REASONING ENGINES        │
   │                               │   │                                 │
   │  DeepResearchPipeline         │   │  ChainOfThought                 │
   │  ├─ Query generation          │   │  ├─ Zero-shot CoT               │
   │  ├─ Iterative search loop     │   │  ├─ Few-shot CoT                │
   │  ├─ Gap analysis              │   │  ├─ Structured CoT              │
   │  ├─ Final synthesis           │   │  └─ Self-consistency            │
   │  └─ SSE streaming             │   │                                 │
   │                               │   │  ParallelSampler                │
   │  MetaCoT                      │   │  ├─ Best-of-N                   │
   │  ├─ <search> action tags      │   │  ├─ Tournament selection        │
   │  ├─ <reading> action tags     │   │  └─ Diversity-aware selection   │
   │  ├─ <think> reasoning tags    │   │                                 │
   │  ├─ <answer> final tags       │   │  SequentialSampler              │
   │  └─ Agentic loop              │   │  ├─ Critique-refine loop        │
   │                               │   │  ├─ Decompose-and-solve         │
   └───────────────┬───────────────┘   │  └─ Debate mode                │
                   │                   │                                 │
                   │                   │  TreeOfThoughts                 │
                   │                   │  ├─ Beam search (recommended)   │
                   │                   │  ├─ BFS                         │
                   │                   │  └─ DFS                         │
                   │                   │                                 │
                   │                   │  InferenceTimeScaler            │
                   │                   │  ├─ SAMPLES strategy            │
                   │                   │  ├─ STEPS strategy              │
                   │                   │  ├─ SEARCH strategy             │
                   │                   │  ├─ REVISION strategy           │
                   │                   │  └─ ADAPTIVE (auto-select)      │
                   │                   │                                 │
                   │                   │  Verifiers                      │
                   │                   │  ├─ OutcomeRewardModel (ORM)    │
                   │                   │  ├─ ProcessRewardModel (PRM)    │
                   │                   │  ├─ RuleBasedVerifier           │
                   │                   │  └─ BestOfNWithVerifier         │
                   │                   └─────────────────────────────────┘
                   │
   ┌───────────────▼───────────────────────────────────────────────────┐
   │                         SEARCH LAYER                               │
   │                                                                    │
   │  FallbackSearchEngine                                              │
   │  ├─ DuckDuckGoProvider  ──┐                                        │
   │  ├─ SerpAPIProvider    ───┼──► SearchAggregator                   │
   │  ├─ BraveProvider     ────┤    ├─ URL deduplication               │
   │  └─ BingProvider      ────┘    ├─ Relevance scoring               │
   │                                ├─ Domain trust weighting           │
   │  ContentExtractor              └─ Token budget distribution        │
   │  ├─ trafilatura   (articles — highest quality)                     │
   │  ├─ readability   (general pages)                                  │
   │  ├─ BeautifulSoup (fallback)                                       │
   │  └─ Playwright    (JavaScript-rendered pages)                      │
   └───────────────┬───────────────────────────────────────────────────┘
                   │
   ┌───────────────▼───────────────────────────────────────────────────┐
   │                     INFRASTRUCTURE LAYER                           │
   │                                                                    │
   │  TokenCounter          RateLimiter         LayeredCache            │
   │  ├─ Per-model limits   ├─ RPM per model    ├─ L1: Disk (fast)     │
   │  ├─ Pre-flight check   ├─ TPM per model    ├─ L2: Redis (shared)  │
   │  ├─ Smart truncation   ├─ Async-safe       └─ TTL + LRU eviction  │
   │  └─ Message trimming   └─ Token bucket                            │
   │                                                                    │
   │  Model Providers                                                   │
   │  ├─ OpenAIModel    (o3, o1, gpt-4o — with o-series handling)      │
   │  ├─ DeepSeekModel  (R1 — with <think> tag parsing)                │
   │  └─ OllamaModel    (local — deepseek-r1:7b through 70b)           │
   └───────────────────────────────────────────────────────────────────┘
```

### Request Data Flow

```
POST /api/v1/research  {"question": "..."}
         │
         ▼
  [1] Validate token budget
         │
         ▼
  [2] Generate N search queries via LLM
         │
         ▼
  [3] Execute queries in parallel
      DuckDuckGo / SerpAPI / Brave / Bing
         │
         ▼
  [4] Aggregate + deduplicate results
      Score by relevance · Trust domain
         │
         ▼
  [5] Fetch full content (top K URLs)
      trafilatura → readability → BS4 → Playwright
         │
         ▼
  [6] Build context string (token-budget-aware)
         │
         ▼
  [7] Gap analysis via LLM
      ├── confidence=high ──────────────────► [8]
      └── confidence=low/medium ─► new queries ─► [3]
         │
         ▼
  [8] Final synthesis via LLM
         │
         ▼
  ResearchReport {answer, sources, confidence, metadata}
```

---

## 🧠 Reasoning Techniques

### 1 · Chain-of-Thought (CoT)

Elicits explicit step-by-step reasoning before producing
a final answer. Three modes are supported:

| Mode | When to Use | Mechanism |
|------|-------------|-----------|
| `zero_shot` | Quick reasoning | Append "Let's think step by step" |
| `few_shot` | Consistent format | Provide worked examples first |
| `structured` | Production use | Enforce `<reasoning>` / `<answer>` tags |

**Self-consistency** samples N independent paths and takes a
majority vote over final answers, improving accuracy by
10–20% on reasoning benchmarks (Wang et al., 2022).

```
Question ──► Path 1: reasoning → answer A ──┐
         ──► Path 2: reasoning → answer A ──┼──► Majority vote → A ✓
         ──► Path 3: reasoning → answer B ──┘
         ──► Path 4: reasoning → answer A ──┘
```

---

### 2 · Parallel Sampling

Generates N independent completions simultaneously and
selects the best using a quality scorer.

```
                    ┌──► Sample 1 → score 0.62 ──┐
                    ├──► Sample 2 → score 0.89 ──┤
Question ──► Model ─┤──► Sample 3 → score 0.74 ──┼──► Best: Sample 2
                    ├──► Sample 4 → score 0.55 ──┤
                    └──► Sample N → score 0.81 ──┘
```

**Selection strategies:**

| Strategy | Description | Cost |
|----------|-------------|:----:|
| `score` | Heuristic quality scorer | Low |
| `tournament` | Pairwise LLM comparison | Medium |
| `diverse` | Quality + diversity balance | Medium |

---

### 3 · Sequential Sampling

Iteratively improves a single answer through structured feedback.

```
Initial Draft
     │
     ▼
 Self-Critique (severity: low / medium / high)
     │
     ├── severity=low ──────────────────────► Final Answer ✓
     │
     └── severity=medium/high
              │
              ▼
         Refined Draft
              │
              ▼
         Self-Critique → ... (up to max_steps)
```

**Variants:**

| Mode | Description |
|------|-------------|
| `refinement` | Critique → refine loop |
| `decompose_and_solve` | Break into sub-tasks, solve sequentially |
| `debate` | Argue for/against, then synthesize |

---

### 4 · Tree of Thoughts (ToT)

Explores a tree of intermediate reasoning steps, scoring and
pruning at each level to find the optimal reasoning path.

```
                    [Root Problem]
                   /      │       \
            [Step A]  [Step B]  [Step C]
            score=0.8 score=0.3 score=0.6
              /  \                  │
          [A1]  [A2]             [C1]
          0.91  0.44             0.72
           │
      [Solution ✓]
      score=0.91
```

**Search strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `beam` | Keep top-K at each depth | Most problems (recommended) |
| `bfs` | Explore all nodes level by level | Short reasoning chains |
| `dfs` | Go deep on most promising path | Long reasoning chains |

---

### 5 · Meta-CoT — Agentic Search

The model itself decides when to search, what to search for,
and integrates retrieved information into its own reasoning trace.
No fixed retrieval step — the model drives the entire process.

```
User: "What specific papers did DeepSeek publish in 2025?"

<think>
  I need to find specific paper titles and dates.
  Let me search for this.
</think>
<search>DeepSeek AI papers published 2025</search>

  [Search results injected automatically]

<think>
  I found DeepSeek-R1 and DeepSeek-V3. I need more
  detail on their specific technical contributions.
</think>
<search>DeepSeek-R1 GRPO training methodology 2025</search>

  [Search results injected automatically]

<think>
  Now I have enough. Let me synthesize.
</think>
<answer>
  DeepSeek published two major papers in early 2025...
  [Full cited answer]
</answer>
```

**Action tags:**

| Tag | Effect |
|-----|--------|
| `<think>...</think>` | Internal reasoning (not shown to user) |
| `<search>query</search>` | Triggers web search |
| `<reading>url</reading>` | Fetches and reads a specific URL |
| `<answer>...</answer>` | Terminates loop, returns final answer |

---

## 📈 Inference-Time Scaling

Spend more compute at inference time — not training time —
to improve answer quality on a per-question basis.

### Strategies

| Strategy | Mechanism | Best For |
|----------|-----------|----------|
| `SAMPLES` | More parallel samples, pick best | Factual questions |
| `STEPS` | Longer sequential refinement | Analytical writing |
| `SEARCH` | Wider / deeper tree search | Math, logic, planning |
| `REVISION` | More self-refinement passes | Nuanced, creative tasks |
| `ADAPTIVE` | Auto-select based on question type | General use |

### Scaling Curve

Quality improves logarithmically as token budget increases:

```
Quality
  1.00 │                                    ●
  0.90 │                          ●
  0.80 │                ●
  0.70 │        ●
  0.60 │  ●
  0.50 └──────────────────────────────────────► Token Budget
        1K     2K      4K      8K     16K    32K
```

### Budget Control

```python
budget = ScalingBudget(
    token_budget   = 16_000,   # Hard token cap
    time_budget_s  = 60.0,     # Hard time cap
    max_samples    = 8,        # Max parallel samples
    max_steps      = 4,        # Max sequential steps
    max_depth      = 4,        # Max tree depth
)

# Auto-scale budget by difficulty
hard_budget = budget.scale_from_difficulty("hard")      # 1× multiplier
easy_budget = budget.scale_from_difficulty("easy")      # 0.25× multiplier
hard_budget = budget.scale_from_difficulty("very_hard") # 2× multiplier
```

### Adaptive Strategy Selection Rules

```
token_budget < 2,000    → SAMPLES  (fast parallel)
question has math/proof → SEARCH   (tree of thoughts)
question starts with    → STEPS    (long reasoning)
  "explain" / "why"
question starts with    → REVISION (polish output)
  "write" / "create"
default                 → SAMPLES  (reliable baseline)
```

---

## 🚀 Quick Start

Choose the path that fits your setup:

### Option A · Docker (Recommended)

```bash
# 1. Clone
git clone https://github.com/yourorg/deep-research.git
cd deep-research

# 2. Configure
cp .env.example .env
# Open .env and add at least one of:
#   OPENAI_API_KEY=sk-...
#   DEEPSEEK_API_KEY=ds-...

# 3. Launch
docker-compose up -d

# 4. Verify
curl http://localhost:8000/api/v1/health

# 5. First research query
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is fusion energy?", "provider": "openai"}'
```

---

### Option B · Local Python

```bash
git clone https://github.com/yourorg/deep-research.git
cd deep-research

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env              # Edit with your keys

uvicorn src.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

### Option C · Fully Local (No API Keys Required)

```bash
# Step 1: Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Step 2: Pull DeepSeek-R1
ollama pull deepseek-r1:7b        # 4.7 GB  ·  needs 8 GB RAM
ollama pull deepseek-r1:32b       # 19  GB  ·  needs 32 GB RAM

# Step 3: Launch stack
docker-compose --profile local up -d

# Step 4: Research (free, offline)
deep-research ask "How does CRISPR work?" --provider ollama
```

---

## 📦 Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|:-------:|:-----------:|
| Python | 3.11 | 3.11+ |
| RAM | 4 GB | 16 GB |
| Disk | 5 GB | 50 GB (local models) |
| GPU | Not required | NVIDIA 24 GB+ (training only) |
| OS | Linux / macOS / Windows | Ubuntu 22.04 LTS |

### Install Variants

```bash
# Base — API + web search only
pip install -r requirements.txt

# With GPU training support
pip install -e ".[training]"

# With local model support (vLLM, llama.cpp)
pip install -e ".[local]"

# Full development environment
pip install -e ".[dev,training,local]"
pre-commit install
```

### Optional: Playwright (JavaScript pages)

```bash
playwright install chromium --with-deps
```

### Verify Installation

```bash
python -c "
from src.models.base_model import GenerationConfig
from src.reasoning.chain_of_thought import ChainOfThought
from src.pipeline.deep_research import DeepResearchPipeline
from src.reasoning.inference_scaling import InferenceTimeScaler
print('✓ All core imports OK')
"
```

---

## ⚙️ Configuration

All settings are loaded from environment variables.
Copy the template and edit:

```bash
cp .env.example .env
```

### LLM Providers (at least one required)

```env
# ── OpenAI ──────────────────────────────────────────
OPENAI_API_KEY=sk-proj-...
OPENAI_ORG_ID=org-...                      # optional
OPENAI_DEFAULT_MODEL=gpt-4o
OPENAI_REASONING_MODEL=o3-mini

# ── DeepSeek ─────────────────────────────────────────
DEEPSEEK_API_KEY=ds-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_REASONING_MODEL=deepseek-reasoner

# ── Local (Ollama) ────────────────────────────────────
LOCAL_MODEL_TYPE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:7b
```

### Search Providers (DuckDuckGo works with no key)

```env
SEARCH_PROVIDER=duckduckgo         # duckduckgo | serpapi | brave | bing
SERPAPI_KEY=                       # optional — Google quality results
BRAVE_SEARCH_API_KEY=              # optional — privacy-focused
BING_SEARCH_API_KEY=               # optional — Microsoft index
MAX_SEARCH_RESULTS=10
SEARCH_TIMEOUT_SECONDS=15
MAX_CONTENT_LENGTH=8000            # tokens per extracted page
```

### Application Behaviour

```env
APP_ENV=production                 # development | staging | production
LOG_LEVEL=INFO                     # DEBUG | INFO | WARNING | ERROR
REDIS_URL=redis://localhost:6379/0
```

### Token & Budget Limits

```env
MAX_TOKENS_PER_REQUEST=128000
REASONING_BUDGET=32768
MAX_OUTPUT_TOKENS=8192
```

### Research Pipeline

```env
MAX_RESEARCH_ITERATIONS=3
MAX_SEARCH_PER_ITERATION=3
```

### Reasoning Defaults

```env
PARALLEL_SAMPLES=3
SEQUENTIAL_STEPS=3
TEMPERATURE=0.7
TOP_P=0.9
TOT_BRANCHING_FACTOR=3
TOT_MAX_DEPTH=4
TOT_BEAM_WIDTH=2
```

### Full `.env.example` Reference

```env
# ── LLM Providers ──────────────────────────────────────────────────────────
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

# ── Search ─────────────────────────────────────────────────────────────────
SEARCH_PROVIDER=duckduckgo
SERPAPI_KEY=
BRAVE_SEARCH_API_KEY=
BING_SEARCH_API_KEY=
MAX_SEARCH_RESULTS=10
SEARCH_TIMEOUT_SECONDS=15
MAX_CONTENT_LENGTH=8000

# ── App ────────────────────────────────────────────────────────────────────
APP_ENV=development
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379/0

# ── Limits ─────────────────────────────────────────────────────────────────
MAX_TOKENS_PER_REQUEST=128000
REASONING_BUDGET=32768
MAX_OUTPUT_TOKENS=8192

# ── Pipeline ───────────────────────────────────────────────────────────────
MAX_RESEARCH_ITERATIONS=3
MAX_SEARCH_PER_ITERATION=3

# ── Reasoning ──────────────────────────────────────────────────────────────
PARALLEL_SAMPLES=3
SEQUENTIAL_STEPS=3
TEMPERATURE=0.7
TOP_P=0.9
TOT_BRANCHING_FACTOR=3
TOT_MAX_DEPTH=4
TOT_BEAM_WIDTH=2
```

---

## 💻 CLI Reference

Install CLI commands:

```bash
pip install -e .
```

Two commands are registered:

| Command | Description |
|---------|-------------|
| `deep-research` | Research, search, and scaling |
| `deep-research-train` | STaR, GRPO, SFT, and pair generation |

---

### `deep-research ask`

Run a research query.

```bash
deep-research ask [QUESTION] [OPTIONS]

Options:
  --provider  -p   openai | deepseek | ollama          [default: openai]
  --model     -m   Model name override
  --mode           deep_research | meta_cot | cot | tot [default: deep_research]
  --iterations -i  Max research iterations  1-10        [default: 3]
  --stream    -s   Stream progress in real time
  --fast      -f   Fast mode: 1 iteration, 2 queries
  --json      -j   Output raw JSON
```

**Examples:**

```bash
# Standard deep research
deep-research ask "What are the latest fusion energy breakthroughs?"

# Use DeepSeek-R1 API
deep-research ask "Explain quantum entanglement" \
  --provider deepseek \
  --model deepseek-reasoner

# Fully local — free and offline
deep-research ask "How does mRNA vaccine work?" \
  --provider ollama \
  --model deepseek-r1:7b

# Stream real-time progress
deep-research ask "History of the internet" --stream

# Agentic mode — model drives its own search
deep-research ask "Latest AI papers 2025" --mode meta_cot

# CoT only — no web search
deep-research ask "Prove there are infinite primes" --mode cot

# Tree of Thoughts
deep-research ask "If 3x + 7 = 22, what is x?" --mode tot

# Fast single-iteration research
deep-research ask "What is GPT-4o?" --fast

# Raw JSON output (pipe to jq)
deep-research ask "What is inflation?" --json | jq '.answer'

# 5 iterations for deep topics
deep-research ask "Origins of the Roman Empire" --iterations 5
```

---

### `deep-research search`

Execute a web search directly.

```bash
deep-research search [QUERY] [OPTIONS]

Options:
  --n      Number of results     [default: 10]
  --fetch  Fetch full page content
```

```bash
# Basic search
deep-research search "fusion energy 2024"

# More results
deep-research search "quantum computing" --n 20

# Fetch full content from top pages
deep-research search "DeepSeek R1 paper" --fetch
```

---

### `deep-research scale`

Run inference-time scaling on a question.

```bash
deep-research scale [QUESTION] [OPTIONS]

Options:
  --provider  -p   openai | deepseek | ollama    [default: openai]
  --model     -m   Model name override
  --budget    -b   Token budget                  [default: 8000]
  --strategy       samples | steps | search | revision | adaptive
  --curve          Show quality vs. compute scaling curve
```

```bash
# Adaptive strategy, 8K budget
deep-research scale "Prove sqrt(2) is irrational" --budget 8000

# Explicit strategy
deep-research scale "Explain transformer architecture" \
  --strategy steps --budget 12000

# Show scaling curve across budgets
deep-research scale "What is quantum computing?" \
  --curve --strategy samples --provider ollama
```

**Scaling curve output:**

```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃  Token Budget   ┃ Quality Score ┃ Strategy ┃ Candidates ┃  Time(s) ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ 1,000           │ 0.612         │ samples  │ 1          │ 2.1      │
│ 2,000           │ 0.698         │ samples  │ 1          │ 2.4      │
│ 4,000           │ 0.741         │ samples  │ 2          │ 3.8      │
│ 8,000           │ 0.803         │ samples  │ 4          │ 6.2      │
│ 16,000          │ 0.851         │ samples  │ 8          │ 11.4     │
│ 32,000          │ 0.889         │ samples  │ 16         │ 21.7     │
└─────────────────┴───────────────┴──────────┴────────────┴──────────┘
```

---

### `deep-research-train star`

Collect STaR reasoning training data.

```bash
deep-research-train star [DATA_PATH] [OPTIONS]

Options:
  --provider    -p  ollama | openai | deepseek   [default: ollama]
  --model       -m  Model name override
  --iterations  -i  STaR iterations             [default: 3]
  --concurrent  -c  Concurrent requests         [default: 3]
  --output      -o  Output JSON path
```

```bash
deep-research-train star data/problems.json \
  --provider ollama \
  --model deepseek-r1:7b \
  --iterations 3 \
  --concurrent 5 \
  --output data/star_dataset.json
```

---

### `deep-research-train sft`

Fine-tune a model on collected STaR data.

```bash
deep-research-train sft [DATASET_PATH] [OPTIONS]

Options:
  --base-model   HuggingFace model ID or local path
  --output   -o  Output directory
  --epochs   -e  Training epochs               [default: 3]
  --batch-size   Batch size per device         [default: 4]
  --lora/--no-lora  Use LoRA (recommended)
  --lora-rank    LoRA rank                     [default: 16]
```

```bash
deep-research-train sft data/star_dataset.json \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --epochs 3 \
  --lora \
  --lora-rank 16 \
  --output models/sft_v1
```

---

### `deep-research-train grpo`

Collect GRPO episodes and run RL training.

```bash
deep-research-train grpo [DATA_PATH] [OPTIONS]

Options:
  --policy       Policy model path / HF ID
  --reference    Reference model (default: same as policy)
  --group-size   GRPO group size G             [default: 8]
  --epochs   -e  Training epochs               [default: 1]
  --output   -o  Output directory
```

```bash
deep-research-train grpo data/problems.json \
  --policy deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --group-size 8 \
  --output models/grpo_v1
```

---

### `deep-research-train pairs`

Generate synthetic preference pairs for reward model training.

```bash
deep-research-train pairs [QUESTIONS_PATH] [OPTIONS]

Options:
  --teacher          Strong judge provider (openai / deepseek)
  --student          Weaker student provider  (ollama)
  --teacher-model    Teacher model name override
  --student-model    Student model name override
  --n-samples        Samples per question      [default: 4]
  --output       -o  Output JSON path
```

```bash
deep-research-train pairs data/questions.txt \
  --teacher openai --teacher-model o3-mini \
  --student ollama --student-model deepseek-r1:7b \
  --n-samples 4 \
  --output data/preference_pairs.json
```

---

## 🐍 Python API

### Deep Research Pipeline

```python
import asyncio
from src.models.local_models import get_model_factory
from src.pipeline.deep_research import DeepResearchPipeline

async def main():
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

    # Markdown-formatted full report
    print(report.to_markdown())

    # Individual fields
    print(f"Confidence:   {report.confidence}")
    print(f"Sources:      {report.source_urls}")
    print(f"Tokens used:  {report.total_tokens_used:,}")
    print(f"Iterations:   {len(report.iterations)}")

asyncio.run(main())
```

### Streaming Research

```python
import asyncio
from src.models.local_models import get_model_factory
from src.pipeline.deep_research import DeepResearchPipeline

async def main():
    model    = get_model_factory("openai", "o3-mini")
    pipeline = DeepResearchPipeline(model=model, max_iterations=2)

    async for event in pipeline.stream_research(
        "What is the current state of quantum computing?"
    ):
        t = event["type"]
        c = event["content"]

        if t == "queries":         print(f"🔍  {c}")
        elif t == "sources_ready": print(f"📄  {c}")
        elif t == "gap_analysis":  print(f"🔬  {c}")
        elif t == "synthesizing":  print(f"⚗️   {c}")
        elif t == "complete":
            print(f"\n📝  Answer:\n{c}")
            meta = event.get("metadata", {})
            print(f"Sources: {meta.get('source_urls', [])}")

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

    # Structured CoT
    result = await cot.generate(
        question=(
            "A store offers 25% off $80, "
            "then 10% off the sale price. Final price?"
        ),
        mode="structured",
        config=GenerationConfig(max_tokens=2048, temperature=0.5),
    )
    print(f"Answer:    {result.answer}")
    print(f"Reasoning: {result.reasoning}")

    # Self-consistency — majority vote over 7 samples
    sc = await cot.self_consistency(
        question="What is the 10th Fibonacci number?",
        n_samples=7,
        mode="structured",
    )
    print(f"Answer:     {sc.answer}")
    print(f"Confidence: {sc.confidence:.0%}")
    print(f"Votes:      {sc.supporting_votes}/{sc.total_votes}")

asyncio.run(main())
```

### Tree of Thoughts

```python
import asyncio
from src.models.local_models import get_model_factory
from src.reasoning.tree_of_thoughts import TreeOfThoughts, SearchStrategy

async def main():
    model = get_model_factory("openai", "o3-mini")
    tot   = TreeOfThoughts(
        model=model,
        branching_factor=3,
        max_depth=4,
        beam_width=2,
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

    # Single scaled answer
    result = await scaler.scale(
        question=(
            "What are the strongest arguments for and "
            "against universal basic income?"
        ),
        budget=ScalingBudget(
            token_budget=16_000,
            max_samples=8,
            max_steps=4,
            time_budget_s=120.0,
        ),
        strategy=ScalingStrategy.ADAPTIVE,
    )

    print(f"Strategy:    {result.strategy_used}")
    print(f"Tokens:      {result.tokens_spent:,}")
    print(f"Quality:     {result.quality_score:.3f}")
    print(f"Time:        {result.time_spent_s:.1f}s")
    print(f"Answer:\n{result.answer}")

    # Scaling curve — quality vs. compute
    curve = await scaler.compute_scaling_curve(
        question="Explain the Riemann Hypothesis",
        token_budgets=[1_000, 2_000, 4_000, 8_000, 16_000],
        strategy=ScalingStrategy.SAMPLES,
    )
    for r in curve:
        print(f"{r.tokens_spent:>7,} tokens → quality {r.quality_score:.3f}")

asyncio.run(main())
```

### Meta-CoT

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
        "What specific papers did DeepSeek publish in 2025 "
        "and what were their main technical contributions?"
    )

    print(f"Answer:\n{result.answer}")
    print(f"\nSearches:  {result.total_searches}")
    print(f"Readings:  {result.total_readings}")
    print(f"Sources:   {result.sources_used}")
    print(f"Tokens:    {result.total_tokens:,}")

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
            "Write a clear, accurate explanation of how "
            "neural networks learn via backpropagation."
        )
    )

    print(f"Refinements:  {trace.n_refinements}")
    print(f"Converged:    {trace.converged}")
    print(f"Score:  {trace.scores[0]:.2f} → {trace.scores[-1]:.2f}  "
          f"(+{trace.improvement:.2f})")
    print(f"\nFinal answer:\n{trace.final_answer}")

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

    # ORM — score a final answer
    orm    = OutcomeRewardModel(judge_model=model)
    result = await orm.verify(
        question="Boiling point of water at sea level?",
        answer="Water boils at 100°C (212°F) at sea level.",
    )
    print(f"ORM score:  {result.score:.2f}")
    print(f"Correct:    {result.is_correct}")
    print(f"Feedback:   {result.feedback}")

    # Rule-based — exact / numeric match
    rule   = RuleBasedVerifier()
    result = await rule.verify(
        question="What is 15% of 80?",
        answer="12",
        reference="12",
    )
    print(f"\nRule score: {result.score}  correct={result.is_correct}")

    # Best-of-N with ORM
    bon = BestOfNWithVerifier(generator=model, verifier=orm)
    best, verification, all_results = await bon.generate_and_verify(
        question="Explain the Pythagorean theorem",
        n=4,
    )
    scores = [f"{v.score:.2f}" for v in all_results]
    print(f"\nAll scores: {scores}")
    print(f"Best score: {verification.score:.2f}")

asyncio.run(main())
```

---

## 🌐 REST API

### Start the Server

```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Production (4 workers)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Interactive docs
open http://localhost:8000/docs
```

### Endpoint Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/research` | Deep research |
| `POST` | `/api/v1/research/stream` | SSE streaming research |
| `POST` | `/api/v1/cot` | Chain-of-Thought |
| `POST` | `/api/v1/tot` | Tree of Thoughts |
| `POST` | `/api/v1/search` | Web search |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

---

### `POST /api/v1/research`

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "question":       "What are the main risks of AI alignment?",
    "provider":       "openai",
    "model_name":     "o3-mini",
    "max_iterations": 3,
    "fast_mode":      false,
    "mode":           "deep_research"
  }'
```

**Request:**

```json
{
  "question":       "string  (5–2000 chars, required)",
  "provider":       "openai | deepseek | ollama",
  "model_name":     "string  (optional)",
  "max_iterations": "integer 1–10  (default: 3)",
  "fast_mode":      "boolean        (default: false)",
  "mode":           "deep_research | meta_cot | cot | tot | self_consistency | parallel"
}
```

**Response:**

```json
{
  "question":           "string",
  "answer":             "string",
  "confidence":         "low | medium | high",
  "sources": [
    {
      "title":           "string",
      "url":             "string",
      "domain":          "string",
      "relevance_score": "float 0.0–1.0",
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

---

### `POST /api/v1/research/stream` (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Latest AI breakthroughs", "provider": "openai"}'
```

**Event stream format:**

```
data: {"type": "start",           "content": "Starting research..."}
data: {"type": "iteration_start", "content": "Research iteration 1",
       "metadata": {"iteration": 1}}
data: {"type": "queries",         "content": "Searching: ...",
       "metadata": {"queries": ["query1", "query2", "query3"]}}
data: {"type": "search_complete", "content": "Found 18 results",
       "metadata": {"total_results": 18}}
data: {"type": "sources_ready",   "content": "Processing 5 sources",
       "metadata": {"sources": [{"title": "...", "url": "..."}]}}
data: {"type": "gap_analysis",    "content": "Confidence: medium | Gaps: 2",
       "metadata": {"confidence": "medium", "gaps": ["gap1", "gap2"]}}
data: {"type": "synthesizing",    "content": "Generating final answer..."}
data: {"type": "complete",        "content": "<full answer text>",
       "metadata": {"confidence": "high", "total_sources": 12,
                    "iterations": 2, "total_time_ms": 18340}}
data: [DONE]
```

---

### `POST /api/v1/cot`

```bash
curl -X POST http://localhost:8000/api/v1/cot \
  -H "Content-Type: application/json" \
  -d '{
    "question":   "If x² - 5x + 6 = 0, what are the values of x?",
    "cot_mode":   "structured",
    "n_samples":  5,
    "provider":   "openai",
    "temperature": 0.7
  }'
```

### `POST /api/v1/tot`

```bash
curl -X POST http://localhost:8000/api/v1/tot \
  -H "Content-Type: application/json" \
  -d '{
    "problem":          "How many ways can 4 people sit at a round table?",
    "branching_factor": 3,
    "max_depth":        4,
    "beam_width":       2,
    "strategy":         "beam",
    "provider":         "openai"
  }'
```

### `POST /api/v1/search`

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query":        "DeepSeek R1 technical report 2025",
    "max_results":  10,
    "fetch_content": false
  }'
```

### `GET /api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status":         "healthy",
  "python_version": "3.11.8",
  "redis":          "connected",
  "timestamp":      1717000000.0
}
```

---

## 🎓 Training Pipeline

Improve the model's reasoning capability over time.
All components are independent — use only what you need.

### Overview

```
① STaR  ──► Collect reasoning examples from the model itself
② SFT   ──► Fine-tune on collected examples (LoRA-efficient)
③ Pairs ──► Generate preference pairs (teacher vs. student)
④ ORM   ──► Train outcome reward model on preference pairs
⑤ GRPO  ──► RL fine-tuning with verifier as reward signal
```

---

### Step 1 · Collect Reasoning Data (STaR)

Prepare `data/problems.json`:

```json
[
  {"question": "What is 15% of 240?",               "answer": "36"},
  {"question": "If 3x + 7 = 22, find x",            "answer": "5"},
  {"question": "What is the capital of Australia?",  "answer": "Canberra"},
  {"question": "Simplify: 2(3x - 4) + 5x",          "answer": "11x - 8"}
]
```

Run STaR:

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
For each problem in dataset:
  ┌─ Generate answer with current model
  │
  ├─ Correct? ──YES──► Save (question, reasoning, answer) as training example
  │
  └─ Wrong? ──► Hint with correct answer ──► Ask model to rationalize
                   │
                   ├─ Rationalization correct? ──YES──► Save as training example
                   │
                   └─ Still wrong? ──► Discard
Repeat for N iterations
```

**Expected output:**

```
STaR Training
  Problems:   1,000
  Iterations: 3
  Provider:   ollama · deepseek-r1:7b

Iteration 1: collected 612/1000  (61.2% accuracy)
Iteration 2: collected 184/388   (47.4% new from rationalization)
Iteration 3: collected  51/204   (25.0% harder remainder)

✓ STaR complete
  Total examples:   847
  Direct:           612
  Rationalized:     235
  Final accuracy:   84.7%
  Saved to:         data/star_dataset.json
```

---

### Step 2 · Fine-Tune on STaR Data (SFT)

Requires GPU. LoRA enables training on consumer hardware.

```bash
deep-research-train sft data/star_dataset.json \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --epochs 3 \
  --batch-size 4 \
  --lora \
  --lora-rank 16 \
  --output models/sft_v1
```

**VRAM requirements with LoRA:**

| Model | VRAM | GPU Example |
|-------|:----:|-------------|
| 1.5B | 6 GB | RTX 3060 |
| 7B | 14 GB | RTX 3090 |
| 14B | 22 GB | RTX 4090 |
| 32B | 48 GB | 2× A100 40GB |
| 70B | 96 GB | 4× A100 40GB |

---

### Step 3 · Generate Preference Pairs

Prepare `data/questions.txt` (one question per line):

```
What is quantum entanglement?
Explain how CRISPR works.
What are the risks of AGI?
How does backpropagation work?
```

```bash
deep-research-train pairs data/questions.txt \
  --teacher openai  --teacher-model o3-mini \
  --student ollama  --student-model deepseek-r1:7b \
  --n-samples 4 \
  --output data/preference_pairs.json
```

---

### Step 4 · RL Training (GRPO)

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
  ① Sample G=8 responses from policy model (parallel)
  ② Score each with verifier → rewards [r₁, r₂, ..., r₈]
  ③ Normalize rewards:
       advantage_i = (r_i − mean(r)) / std(r)
  ④ Compute loss:
       L = −advantage × log_prob + λ·KL(policy ∥ reference)
  ⑤ Update policy weights with gradient descent
```

---

### Training Python API

```python
import asyncio
from src.training.star_trainer import STaRTrainer, STaRConfig
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
    print(f"Examples:  {len(dataset.examples)}")
    print(f"Accuracy:  {dataset.accuracy:.1%}")
    print(f"Direct:    {dataset.n_direct}")
    print(f"Rational.: {dataset.n_rationalized}")

asyncio.run(main())
```

---

## 🖥 Local Deployment

### Quick Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model (choose based on hardware)
ollama pull deepseek-r1:7b    # 8 GB  RAM minimum
ollama pull deepseek-r1:14b   # 16 GB RAM
ollama pull deepseek-r1:32b   # 32 GB RAM
ollama pull deepseek-r1:70b   # 64 GB RAM

# Start API stack with local Ollama
docker-compose --profile local up -d

# Verify
deep-research ask "Explain transformers" \
  --provider ollama --model deepseek-r1:7b
```

### Model Selection Guide

| Use Case | Model | RAM | Speed | Quality |
|----------|-------|:---:|:-----:|:-------:|
| Testing / CI | deepseek-r1:7b | 8 GB | Fast | Good |
| Daily use | deepseek-r1:14b | 16 GB | Medium | Very Good |
| Production | deepseek-r1:32b | 32 GB | Medium | Excellent |
| Research | deepseek-r1:70b | 64 GB | Slow | Best |

### Fully Offline Setup

No API keys, no internet for inference:

```env
# .env — minimal offline config
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

## 🚢 Production Deployment

### Docker Compose Profiles

```bash
# API + Redis  (minimum production)
docker-compose up -d

# + Local Ollama
docker-compose --profile local up -d

# + Monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Full stack
docker-compose --profile local --profile monitoring up -d
```

### Production Checklist

```bash
# ① Harden environment
APP_ENV=production
LOG_LEVEL=WARNING

# ② Redis password
REDIS_URL=redis://:YourStrongPassword@redis:6379/0

# ③ Restrict CORS  (src/api/main.py)
allow_origins=["https://yourdomain.com"]

# ④ HTTPS  — use nginx or Traefik reverse proxy

# ⑤ Health check
curl http://localhost:8000/api/v1/health

# ⑥ Check metrics
curl http://localhost:8000/metrics | grep deep_research
```

### Nginx Configuration (SSE-Critical)

```nginx
server {
    listen 443 ssl http2;
    server_name research.yourdomain.com;

    ssl_certificate      /etc/ssl/certs/cert.pem;
    ssl_certificate_key  /etc/ssl/private/key.pem;

    # SSE streaming endpoint — buffering must be disabled
    location /api/v1/research/stream {
        proxy_pass             http://localhost:8000;
        proxy_http_version     1.1;
        proxy_set_header       Connection '';
        proxy_buffering        off;
        proxy_cache            off;
        proxy_read_timeout     300s;
        chunked_transfer_encoding on;
    }

    # All other endpoints
    location / {
        proxy_pass           http://localhost:8000;
        proxy_set_header     Host $host;
        proxy_set_header     X-Real-IP $remote_addr;
        proxy_set_header     X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout   120s;
    }
}
```

### Kubernetes (Helm values)

```yaml
replicaCount: 3

image:
  repository: yourregistry/deep-research
  tag: "1.0.0"
  pullPolicy: IfNotPresent

env:
  APP_ENV: production
  LOG_LEVEL: INFO
  REDIS_URL: redis://redis-service:6379/0

envFromSecret:
  - secretName: deep-research-secrets
    keys:
      - OPENAI_API_KEY
      - DEEPSEEK_API_KEY
      - SERPAPI_KEY

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
  auth:
    enabled: true
```

---

## 📊 Monitoring

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Prometheus  →  http://localhost:9090
# Grafana     →  http://localhost:3000  (admin / admin)
# Metrics     →  http://localhost:8000/metrics
```

### Available Metrics

```
# Total requests by method, endpoint, status code
deep_research_requests_total{method, endpoint, status_code}

# Request latency histogram
deep_research_request_duration_seconds{endpoint}

# Token consumption by model and type
deep_research_tokens_total{model, type}
```

### Useful PromQL Queries

```promql
# Request rate (last 5 minutes)
rate(deep_research_requests_total[5m])

# P95 response latency
histogram_quantile(0.95,
  rate(deep_research_request_duration_seconds_bucket[5m])
)

# Error rate
rate(deep_research_requests_total{status_code=~"5.."}[5m])
  /
rate(deep_research_requests_total[5m])

# Token spend rate
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
pytest tests/test_models.py               -v   # Model providers + token counting
pytest tests/test_reasoning.py            -v   # CoT, ToT, parallel, sequential
pytest tests/test_search.py               -v   # Search providers + extraction
pytest tests/test_pipeline.py             -v   # Deep research + Meta-CoT
pytest tests/test_inference_scaling.py    -v   # Scaling strategies + budget
```

### With Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
open htmlcov/index.html
```

### Run a Single Test

```bash
pytest tests/test_reasoning.py \
  ::TestChainOfThought::test_self_consistency_majority_vote -v
```

### Expected Coverage

```
Module                                    Stmts   Coverage
──────────────────────────────────────────────────────────
src/models/base_model.py                    89      98%
src/models/openai_models.py                142      91%
src/models/deepseek_models.py              118      89%
src/models/local_models.py                 134      87%
src/reasoning/chain_of_thought.py          198      95%
src/reasoning/parallel_sampling.py         167      93%
src/reasoning/sequential_sampling.py       201      91%
src/reasoning/tree_of_thoughts.py          243      89%
src/reasoning/verifier.py                  189      94%
src/reasoning/inference_scaling.py         178      92%
src/search/web_search.py                   231      88%
src/search/content_extractor.py            198      85%
src/search/search_aggregator.py            167      91%
src/pipeline/deep_research.py              289      87%
src/pipeline/meta_cot.py                   234      88%
src/training/star_trainer.py               198      83%
src/training/reward_models.py              167      81%
src/training/rl_trainer.py                 189      79%
src/training/self_refinement.py            156      92%
src/utils/token_counter.py                 134      98%
src/utils/rate_limiter.py                   89      96%
src/utils/cache.py                         112      93%
──────────────────────────────────────────────────────────
TOTAL                                     3706      90%
```

---

## 📁 Project Structure

```
deep_research/                          ← Project root
│
├── README.md                           ← This file
├── requirements.txt                    ← All runtime dependencies
├── setup.py                            ← Package install + CLI entry points
├── .env.example                        ← Environment variable template
├── .gitignore
├── Dockerfile                          ← Multi-stage: production + development
├── docker-compose.yml                  ← Full stack: API, Redis, Ollama, Grafana
│
├── config/                             ← Application configuration
│   ├── __init__.py
│   ├── settings.py                     ← Pydantic BaseSettings (env vars)
│   ├── logging_config.py               ← Loguru structured logging setup
│   └── prometheus.yml                  ← Prometheus scrape configuration
│
├── src/                                ← All application source code
│   ├── __init__.py
│   │
│   ├── models/                         ← LLM provider clients
│   │   ├── __init__.py                 ← Clean exports + get_model_factory
│   │   ├── base_model.py               ← Abstract interface: BaseModel,
│   │   │                               │  GenerationConfig, ModelResponse,
│   │   │                               │  ModelCapability
│   │   ├── openai_models.py            ← OpenAI: o3, o1, gpt-4o
│   │   │                               │  Handles o-series system msg conversion,
│   │   │                               │  reasoning_effort, reasoning_tokens
│   │   ├── deepseek_models.py          ← DeepSeek: deepseek-reasoner (R1)
│   │   │                               │  Parses <think>...</think> tags
│   │   └── local_models.py             ← Ollama: deepseek-r1 7b/14b/32b/70b
│   │                                   │  Auto-pull, context limit enforcement,
│   │                                   │  model factory function
│   │
│   ├── reasoning/                      ← Inference-time reasoning engines
│   │   ├── __init__.py                 ← Exports all reasoning classes
│   │   ├── chain_of_thought.py         ← Zero-shot / few-shot / structured CoT
│   │   │                               │  + self-consistency with majority vote
│   │   ├── parallel_sampling.py        ← Best-of-N parallel sampling
│   │   │                               │  Tournament + diversity-aware selection
│   │   ├── sequential_sampling.py      ← Critique-refine loop
│   │   │                               │  Decompose-and-solve + Debate mode
│   │   ├── tree_of_thoughts.py         ← Full ToT: ThoughtNode tree,
│   │   │                               │  BFS / DFS / Beam search strategies,
│   │   │                               │  LLM-scored node evaluation
│   │   ├── verifier.py                 ← OutcomeRewardModel (ORM)
│   │   │                               │  ProcessRewardModel (PRM, step-level)
│   │   │                               │  RuleBasedVerifier (exact + numeric)
│   │   │                               │  BestOfNWithVerifier
│   │   └── inference_scaling.py        ← Unified inference-time scaling:
│   │                                   │  ScalingBudget, ScalingStrategy,
│   │                                   │  ScalingResult, InferenceTimeScaler,
│   │                                   │  scaling curve computation
│   │
│   ├── search/                         ← Web search and content extraction
│   │   ├── __init__.py                 ← Exports search_engine, extractor
│   │   ├── web_search.py               ← Four providers with unified interface:
│   │   │                               │  DuckDuckGo (free), SerpAPI, Brave,
│   │   │                               │  Bing + FallbackSearchEngine
│   │   │                               │  SearchResult, SearchResponse schemas
│   │   ├── content_extractor.py        ← URL content extraction cascade:
│   │   │                               │  trafilatura → readability →
│   │   │                               │  BeautifulSoup → Playwright
│   │   │                               │  Token-aware truncation, caching,
│   │   │                               │  blocked domain / extension filtering
│   │   └── search_aggregator.py        ← Multi-response aggregation:
│   │                                   │  URL deduplication, relevance scoring,
│   │                                   │  domain trust weighting,
│   │                                   │  token-budget context builder
│   │
│   ├── pipeline/                       ← High-level research orchestrators
│   │   ├── __init__.py
│   │   ├── deep_research.py            ← DeepResearchPipeline:
│   │   │                               │  iterative search-reason-gap loop,
│   │   │                               │  ResearchReport with to_markdown(),
│   │   │                               │  SSE streaming via stream_research()
│   │   └── meta_cot.py                 ← MetaCoT agentic pipeline:
│   │                                   │  <search> / <reading> / <think> /
│   │                                   │  <answer> tag parsing and execution,
│   │                                   │  streaming via stream_reason()
│   │
│   ├── training/                       ← Model training pipelines
│   │   ├── __init__.py
│   │   ├── star_trainer.py             ← STaRTrainer: direct + rationalization
│   │   │                               │  collection, STaRDataset with
│   │   │                               │  HF Dataset export, SFTTrainer
│   │   │                               │  wrapper (LoRA via PEFT + TRL)
│   │   ├── reward_models.py            ← SyntheticPreferenceGenerator,
│   │   │                               │  PreferencePair schema,
│   │   │                               │  RewardModelTrainer with
│   │   │                               │  Bradley-Terry pairwise loss
│   │   ├── rl_trainer.py               ← GRPOCollector: episode sampling,
│   │   │                               │  group-relative advantage,
│   │   │                               │  shaped reward function;
│   │   │                               │  GRPOTrainer: KL-penalized update
│   │   └── self_refinement.py          ← SelfRefinementEngine:
│   │                                   │  critique → refine loop,
│   │                                   │  RefinementTrace with score tracking,
│   │                                   │  SelfRefinementConfig
│   │
│   ├── api/                            ← FastAPI web application
│   │   ├── __init__.py
│   │   ├── main.py                     ← App factory: CORS, GZip, request ID
│   │   │                               │  middleware, Prometheus metrics,
│   │   │                               │  global exception handler, lifespan
│   │   ├── routes.py                   ← All route handlers:
│   │   │                               │  /research, /research/stream,
│   │   │                               │  /cot, /tot, /search, /health
│   │   └── schemas.py                  ← Pydantic v2 request / response models:
│   │                                   │  ResearchRequest, ResearchResponse,
│   │                                   │  CoTRequest, CoTResponse,
│   │                                   │  ToTRequest, ToTResponse,
│   │                                   │  SearchRequest, SearchResponse,
│   │                                   │  ErrorResponse
│   │
│   └── utils/                          ← Shared infrastructure utilities
│       ├── __init__.py
│       ├── token_counter.py            ← TokenCounter: per-model encoding,
│       │                               │  pre-flight context validation,
│       │                               │  smart text + message truncation,
│       │                               │  MODEL_CONTEXT_LIMITS map
│       ├── rate_limiter.py             ← TokenBucketRateLimiter: async-safe,
│       │                               │  per-model RPM + TPM enforcement,
│       │                               │  sliding window with backoff
│       └── cache.py                    ← LayeredCache: L1 diskcache (fast,
│                                       │  local), L2 Redis (shared, persistent),
│                                       │  TTL control, graceful Redis fallback
│
├── scripts/                            ← CLI entry points
│   ├── cli.py                          ← deep-research ask / search / scale
│   │                                   │  Rich-formatted terminal output,
│   │                                   │  streaming progress display
│   └── train.py                        ← deep-research-train star / sft /
│                                       │  grpo / pairs
│
├── tests/                              ← Full test suite (pytest + asyncio)
│   ├── __init__.py
│   ├── test_models.py                  ← TokenCounter edge cases,
│   │                                   │  OpenAI o-series message conversion,
│   │                                   │  context overflow validation,
│   │                                   │  ModelResponse defaults
│   ├── test_reasoning.py               ← CoT modes + self-consistency voting,
│   │                                   │  ParallelSampler scoring + similarity,
│   │                                   │  ToT thought parsing + solve mock,
│   │                                   │  ORM / PRM / rule-based verifier
│   ├── test_search.py                  ← Search provider mocking,
│   │                                   │  content extraction strategies,
│   │                                   │  aggregator dedup + ranking
│   ├── test_pipeline.py                ← Full pipeline run with mocked search,
│   │                                   │  query generation, gap analysis,
│   │                                   │  Meta-CoT action parsing + termination,
│   │                                   │  ResearchReport markdown generation
│   └── test_inference_scaling.py       ← ScalingBudget difficulty scaling,
│                                       │  strategy auto-selection rules,
│                                       │  scale_by_samples correctness,
│                                       │  efficiency metric calculation
│
├── data/                               ← Training and evaluation data
│   └── .gitkeep                        ← (git-ignored at runtime)
│
├── models/                             ← Local fine-tuned model checkpoints
│   └── .gitkeep                        ← (git-ignored at runtime)
│
├── logs/                               ← Rotated application logs
│   └── .gitkeep                        ← (git-ignored at runtime)
│
└── notebooks/
    └── demo.ipynb                      ← Interactive end-to-end walkthrough
```

---

## 🗺 Roadmap

### ✅ v1.0.0 — Current

- [x] Multi-provider LLM support (OpenAI o3, DeepSeek-R1, Ollama)
- [x] All inference-time scaling techniques (samples, steps, search, revision, adaptive)
- [x] Full Deep Research pipeline with gap analysis and SSE streaming
- [x] Meta-CoT agentic search with action tags
- [x] Chain-of-Thought: zero-shot / few-shot / structured / self-consistency
- [x] Parallel sampling: best-of-N, tournament, diverse
- [x] Sequential sampling: critique-refine, decompose-solve, debate
- [x] Tree of Thoughts: BFS / DFS / Beam search
- [x] Verifiers: ORM, PRM (step-level), rule-based, best-of-N
- [x] STaR training loop + HuggingFace SFT wrapper
- [x] GRPO RL training with shaped rewards
- [x] Synthetic preference pair generation
- [x] Self-refinement engine
- [x] Production FastAPI with SSE + Prometheus
- [x] Token-safe everywhere (pre-flight validation + smart truncation)
- [x] Redis + disk two-layer cache
- [x] Docker + docker-compose (local, monitoring profiles)
- [x] Rich CLI with streaming support
- [x] 90%+ test coverage

### 🔄 v1.1.0 — In Progress

- [ ] PDF and academic paper ingestion (arXiv, PubMed)
- [ ] Vector store integration (FAISS / Pinecone / Chroma)
- [ ] Multi-modal support — images in research context
- [ ] API key authentication (JWT / API key header)
- [ ] Per-user rate limiting

### 📋 v1.2.0 — Planned

- [ ] Persistent research sessions (save / resume)
- [ ] Custom tool plugins (calculator, code executor, Python REPL)
- [ ] Multi-agent collaboration (researcher + critic + synthesizer)
- [ ] Citation verification system
- [ ] React web UI with streaming research view

### 🔮 v2.0.0 — Future

- [ ] Full RAG pipeline (chunking, embedding, retrieval)
- [ ] Cross-language research (auto-translate + synthesize)
- [ ] Continuous learning from user feedback
- [ ] Domain-specific fine-tuned model hub (medical, legal, scientific)
- [ ] Voice input / output interface

---

## 🤝 Contributing

All contributions are welcome — bug fixes, new features,
documentation, tests, and translations.

### Setup

```bash
# Fork, then clone your fork
git clone https://github.com/YOUR_USERNAME/deep-research.git
cd deep-research

# Create a feature branch
git checkout -b feature/your-feature-name

# Install with dev dependencies
pip install -e ".[dev]"
pre-commit install
```

### Development Workflow

```bash
# Make your changes, then:

# Format
black src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/ --fix

# Type check
mypy src/ --ignore-missing-imports

# Test
pytest tests/ -v --cov=src

# All checks at once
make lint test           # if using Makefile
```

### Contribution Areas

| Area | What We Need |
|------|-------------|
| 🐛 Bug fixes | Regression test with every fix |
| ✨ New reasoning techniques | Benchmark comparison vs. existing |
| 🔍 New search providers | Implement `BaseSearchProvider` interface |
| 🏋️ Better reward models | Trained ORM/PRM to replace heuristic scorer |
| 🧪 Test coverage | Especially search + training modules |
| 📚 Documentation | Tutorials, worked examples, translations |
| 🎨 Web UI | React frontend for the streaming API |

### Pull Request Guidelines

1. One feature or fix per PR
2. All existing tests must pass
3. New features require new tests
4. Update this README if you add CLI commands or new modes
5. Reference the issue number in your PR description
6. Keep commits atomic and messages descriptive

---

## 🙏 Acknowledgements

This system builds on foundational research from the AI community:

| Paper | Authors | Year | Contribution |
|-------|---------|:----:|-------------|
| Chain-of-Thought Prompting | Wei et al. | 2022 | CoT foundation |
| Self-Consistency | Wang et al. | 2022 | Majority-vote sampling |
| STaR: Self-Taught Reasoner | Zelikman et al. | 2022 | Reasoning data bootstrap |
| Tree of Thoughts | Yao et al. | 2023 | ToT search framework |
| Let's Verify Step by Step | Lightman et al. | 2023 | PRM step-level scoring |
| Self-Refine | Madaan et al. | 2023 | Iterative self-improvement |
| Scaling LLM Test-Time Compute | Snell et al. | 2024 | Inference scaling theory |
| OpenAI o1 System Card | OpenAI | 2024 | Reasoning model design |
| DeepSeek-R1 | DeepSeek AI | 2025 | GRPO + reasoning distillation |

**Open-source libraries used:**

[FastAPI](https://fastapi.tiangolo.com) ·
[Pydantic](https://docs.pydantic.dev) ·
[Loguru](https://github.com/Delgan/loguru) ·
[trafilatura](https://trafilatura.readthedocs.io) ·
[duckduckgo-search](https://github.com/deedy5/duckduckgo_search) ·
[tiktoken](https://github.com/openai/tiktoken) ·
[HuggingFace TRL](https://github.com/huggingface/trl) ·
[PEFT](https://github.com/huggingface/peft) ·
[Ollama](https://ollama.ai) ·
[Redis](https://redis.io) ·
[Prometheus](https://prometheus.io)

---

## 📄 License

```
MIT License

Copyright (c) 2025 Deep Research AI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

---

<div align="center">

**Built with ❤️ for the open-source AI research community**

⭐ **Star this repo** if it helped your work

<br/>

[🐛 Report a Bug](https://github.com/yourorg/deep-research/issues/new?template=bug_report.md) ·
[✨ Request a Feature](https://github.com/yourorg/deep-research/issues/new?template=feature_request.md) ·
[💬 Join the Discussion](https://github.com/yourorg/deep-research/discussions) ·
[🐦 Follow Updates](https://twitter.com/yourhandle)

</div>
