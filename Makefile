# ──────────────────────────────────────────────────────────────────────────────
# Deep Research 104 — Makefile
# Common dev / CI commands. Target date: 17 October 2026.
# Usage: make <target>
# ──────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Config ──────────────────────────────────────────────────────────────────
PYTHON      ?= python3
VENV        ?= .venv
PORT        ?= 8000
WORKERS     ?= 4
APP_ENV     ?= development
LOG_LEVEL   ?= info

# ── Helpers ──────────────────────────────────────────────────────────────────
define header
	@echo ""
	@echo "── Deep Research 104 — $(1) ──────────────────────────────────"
endef

.PHONY: help
help:  ## Show this help
	@echo "Deep Research 104 — common commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup           Install dev dependencies in current Python env"
	@echo "  make venv            Create .venv and install dev deps"
	@echo "  make precommit       Install pre-commit hooks"
	@echo ""
	@echo "Run:"
	@echo "  make run             Run API locally (dev, reload)"
	@echo "  make run-prod        Run API in production mode (4 workers)"
	@echo "  make docker-up       Start full stack via docker-compose"
	@echo "  make docker-down     Stop full stack"
	@echo ""
	@echo "Quality:"
	@echo "  make lint            Run ruff + black --check"
	@echo "  make format          Auto-format with ruff + black"
	@echo "  make typecheck       Run mypy on src/"
	@echo "  make test            Run pytest (unit tests only)"
	@echo "  make test-all        Run pytest (incl. slow + integration)"
	@echo "  make coverage        Run tests with coverage report"
	@echo "  make security        Run pip-audit on requirements"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean           Remove caches, build artifacts, venv"
	@echo "  make docs            Serve docs locally (mkdocs)"
	@echo ""

# ── Setup ───────────────────────────────────────────────────────────────────
.PHONY: setup venv precommit

setup:  ## Install dev dependencies in current Python env
	$(call header,Setup)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

venv:  ## Create .venv and install dev deps
	$(call header,Venv)
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements-dev.txt
	$(VENV)/bin/pip install -e .
	@echo "Activate with: source $(VENV)/bin/activate"

precommit:  ## Install pre-commit hooks
	$(call header,Pre-commit)
	pre-commit install
	pre-commit run --all-files || true

# ── Run ─────────────────────────────────────────────────────────────────────
.PHONY: run run-prod docker-up docker-down docker-logs

run:  ## Run API locally (dev, reload)
	$(call header,Run dev)
	APP_ENV=$(APP_ENV) uvicorn src.api.main:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--reload \
		--log-level debug

run-prod:  ## Run API in production mode (4 workers)
	$(call header,Run prod)
	APP_ENV=production uvicorn src.api.main:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--workers $(WORKERS) \
		--log-level $(LOG_LEVEL) \
		--proxy-headers \
		--no-server-header

docker-up:  ## Start full stack via docker-compose
	$(call header,Docker up)
	docker-compose up -d --build

docker-down:  ## Stop full stack
	$(call header,Docker down)
	docker-compose down

docker-logs:  ## Tail docker logs
	docker-compose logs -f --tail=200

# ── Quality ──────────────────────────────────────────────────────────────────
.PHONY: lint format typecheck test test-all coverage security

lint:  ## Run ruff + black --check
	$(call header,Lint)
	ruff check src config tests
	ruff format --check src config tests
	black --check src config tests

format:  ## Auto-format with ruff + black
	$(call header,Format)
	ruff check --fix src config tests
	ruff format src config tests
	black src config tests

typecheck:  ## Run mypy on src/
	$(call header,Typecheck)
	mypy src

test:  ## Run pytest (unit tests only)
	$(call header,Test)
	pytest tests/ -v --tb=short -m "not slow and not integration"

test-all:  ## Run pytest (incl. slow + integration)
	$(call header,Test all)
	pytest tests/ -v --tb=short

coverage:  ## Run tests with coverage report
	$(call header,Coverage)
	pytest tests/ -v --cov=src --cov=config \
		--cov-report=term-missing --cov-report=html:htmlcov \
		-m "not slow and not integration"
	@echo "Open htmlcov/index.html to view the coverage report."

security:  ## Run pip-audit on requirements
	$(call header,Security)
	pip-audit -r requirements.txt --strict
	pip-audit -r requirements-dev.txt --strict

# ── Maintenance ──────────────────────────────────────────────────────────────
.PHONY: clean docs

clean:  ## Remove caches, build artifacts, venv
	$(call header,Clean)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist *.egg-info

docs:  ## Serve docs locally (mkdocs)
	$(call header,Docs)
	mkdocs serve
