# Deep Research 104

> **Production-grade autonomous research engine** combining web search, reasoning models, and inference-time scaling.
>
> **Build date:** 17 October 2026
> **Maintainer:** Adil Shamim (`@AdilShamim8`)
> **License:** MIT

Welcome to the Deep Research 104 documentation site. This site is
built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Quick links

- [Architecture](../ARCHITECTURE.md) — module map, design choices, failure modes
- [Deployment guide](../DEPLOYMENT.md) — Docker, VPS, Kubernetes, TLS, alerts
- [Security policy](../SECURITY.md) — vulnerability reporting + hardening checklist
- [Contributing](../CONTRIBUTING.md) — dev workflow, code style, PR checklist
- [Changelog](../CHANGELOG.md) — Keep a Changelog 1.1.0 + SemVer 2.0
- [Tutorial](../Tutorial-104.md) — long-form explanation of each reasoning technique

## TL;DR

```bash
git clone https://github.com/AdilShamim8/Deep-Research-104.git
cd Deep-Research-104
make venv
source .venv/bin/activate
cp .env.example .env  # then edit .env
make run
# → http://localhost:8000/docs
```

Or, run the full stack via Docker:

```bash
make docker-up
curl http://localhost:8000/health
```

## What's here?

This is the documentation root. The actual content lives in the
repository's top-level Markdown files (linked above) which MkDocs
serves at this URL. To run the docs locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```
