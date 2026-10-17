"""
Unit tests for the deep-research CLI (Typer commands).

Uses Typer's CliRunner to invoke commands without spawning a subprocess.
All commands that would touch the network are stubbed or short-circuited
via env config.
"""

from __future__ import annotations

import pytest
from src.cli.app import app
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    # Click 8.1+ removed the `mix_stderr` kwarg — just use the default.
    return CliRunner()


# ── version ────────────────────────────────────────────────────────────────────


class TestVersionCommand:
    def test_version_prints_metadata(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "deep-research-104" in result.stdout
        assert "1.0.0" in result.stdout
        assert "2026-10-17" in result.stdout


# ── providers ─────────────────────────────────────────────────────────────────


class TestProvidersCommand:
    def test_providers_lists_known_providers(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        for provider in ["openai", "deepseek", "ollama", "vllm", "llamacpp"]:
            assert provider in result.stdout


# ── search ─────────────────────────────────────────────────────────────────────


class TestSearchCommand:
    def test_search_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "--max-results" in result.stdout
        assert "--json" in result.stdout


# ── ask ───────────────────────────────────────────────────────────────────────


class TestAskCommand:
    def test_ask_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.stdout
        assert "--max-iterations" in result.stdout
        assert "--fast-mode" in result.stdout
        assert "--json" in result.stdout


# ── cot ───────────────────────────────────────────────────────────────────────


class TestCotCommand:
    def test_cot_help_shows_modes(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["cot", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.stdout
        assert "--n-samples" in result.stdout
        assert "--temperature" in result.stdout


# ── tot ───────────────────────────────────────────────────────────────────────


class TestTotCommand:
    def test_tot_help_shows_strategies(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["tot", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.stdout
        assert "beam" in result.stdout
        assert "--branching-factor" in result.stdout


# ── serve ─────────────────────────────────────────────────────────────────────


class TestServeCommand:
    def test_serve_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--workers" in result.stdout
        assert "--reload" in result.stdout


# ── Top-level help ────────────────────────────────────────────────────────────


class TestTopLevelHelp:
    def test_no_args_prints_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, [])
        # Typer's no_args_is_help=True should exit with code 0 (or 2)
        # and print the help text.
        assert "Usage" in result.stdout
        # List all commands
        for cmd in ["version", "ask", "cot", "tot", "search", "serve", "providers"]:
            assert cmd in result.stdout
