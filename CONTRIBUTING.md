# Contributing to Deep Research 104

Thank you for your interest in improving **Deep Research 104**! This
document describes how to set up a development environment, run the
tests, and submit changes that match the project's production-grade
standards.

> **Build target date:** 17 October 2026
> **Maintainer:** Adil Shamim (`@AdilShamim8`)
> **License:** MIT (see [LICENSE](LICENSE))

## Table of contents
- [Quick start](#quick-start)
- [Development workflow](#development-workflow)
- [Code style](#code-style)
- [Testing](#testing)
- [Commit message conventions](#commit-message-conventions)
- [Pull request checklist](#pull-request-checklist)
- [Reporting security issues](#reporting-security-issues)

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/AdilShamim8/Deep-Research-104.git
cd Deep-Research-104

# 2. Create a venv and install dev dependencies
make venv
source .venv/bin/activate

# 3. Install pre-commit hooks (runs ruff/black/mypy on every commit)
make precommit

# 4. Copy the env template and fill in any keys you want to test
cp .env.example .env
# (edit .env as needed)

# 5. Run the API locally with hot reload
make run
# → http://localhost:8000/docs
```

If you just want to run the full stack (API + Redis + Prometheus +
Grafana) via Docker:

```bash
make docker-up                  # api + redis (default profile)
make docker-up -- --profile local      # + ollama
make docker-up -- --profile monitoring # + prometheus + grafana
```

## Development workflow

1. **Open an issue first** for anything beyond a typo. This avoids
   wasted effort on changes we can't merge.
2. **Branch off `main`:** `git checkout -b feat/my-feature`.
3. **Keep PRs small and focused** — one logical change per PR is much
   easier to review than a mega-PR.
4. **Run `make lint` and `make test`** before pushing. CI will catch
   the rest, but local runs are faster feedback.
5. **Add tests** for any new behavior. We don't require 100% coverage
   on the diff, but untested new code will be flagged in review.
6. **Document new env vars** in `.env.example` and `config/settings.py`.
7. **Update `README.md`** if your change affects the public API.

## Code style

- **Python 3.11+** — we use `from __future__ import annotations` for
  forward-compat typing.
- **Formatting:** `ruff format` (defaults; line length 100). Do not
  reformat unrelated code in your PR.
- **Linting:** `ruff check` and `black --check` must pass. `mypy` is
  run in best-effort mode and will not block CI, but new code should
  be typed.
- **Imports:** stdlib → third-party → local, separated by a blank line.
- **Docstrings:** every module starts with a `"""..."""` docstring
  that includes a `Date tag: 17 October 2026` line (or the date the
  module was last meaningfully touched). Public functions and classes
  get a docstring; private helpers can skip it.
- **Async-first:** all I/O paths are async. Don't introduce sync HTTP
  calls inside the FastAPI app — use `httpx.AsyncClient` or
  `aiohttp.ClientSession`.
- **Settings:** never read `os.environ` directly. Add the var to
  `config/settings.py` so it gets validated by Pydantic.

## Testing

We use `pytest` + `pytest-asyncio` + `pytest-cov`. Tests live under
`tests/` and are marked:

- `@pytest.mark.unit` — pure unit tests (no network, no services)
- `@pytest.mark.integration` — require Redis, OpenAI, or Ollama
- `@pytest.mark.slow` — take more than ~5s

CI runs only **unit** tests by default. Integration tests run nightly
and on `main` pushes.

```bash
make test           # unit only (fast)
make test-all       # everything
make coverage       # unit + HTML coverage report
```

## Commit message conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

- **type:** `feat`, `fix`, `chore`, `docs`, `refactor`, `perf`,
  `test`, `build`, `ci`, `style`
- **scope:** module name (e.g. `models`, `search`, `api`, `config`)
- **subject:** imperative, lowercase, no trailing period, ≤ 72 chars

Example:

```
fix(search): dedup by normalized URL to avoid cross-provider dupes

The aggregator was treating the same article at www.example.com and
example.com as different sources. Normalize URLs before deduping.

Refs: production hardening phase 3/11
```

## Pull request checklist

- [ ] Branch is up to date with `main`
- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] New code has tests
- [ ] New env vars documented in `.env.example`
- [ ] `README.md` updated if API surface changed
- [ ] Commit messages follow Conventional Commits
- [ ] PR description explains *why*, not just *what*

## Reporting security issues

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email **adilshamim8@users.noreply.github.com** with a
description of the issue and reproduction steps. We will acknowledge
within 48 hours and aim for a fix or mitigation within 14 days.

Thank you for contributing!
