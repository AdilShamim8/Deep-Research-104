"""
Unit tests for the deep-research CLI (Typer commands).

Uses Typer's CliRunner to invoke commands without spawning a subprocess.
All commands that would touch the network are stubbed or short-circuited
via env config.

Note: Rich (Typer's default renderer) emits ANSI color codes in
environments where FORCE_COLOR=1 is set (e.g. CI). We strip those codes
before asserting so the tests are colorblind.
"""

from __future__ import annotations

import re

import pytest
from src.cli.app import app
from typer.testing import CliRunner

# Strip ANSI escape sequences so assertions match the plain text
# even when Rich renders help output with colors.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


@pytest.fixture
def runner() -> CliRunner:
    # Click 8.1+ removed the `mix_stderr` kwarg — just use the default.
    return CliRunner()


# ── version ────────────────────────────────────────────────────────────────────


class TestVersionCommand:
    def test_version_prints_metadata(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "deep-research-104" in out
        assert "1.0.0" in out
        assert "2026-10-17" in out


# ── providers ─────────────────────────────────────────────────────────────────


class TestProvidersCommand:
    def test_providers_lists_known_providers(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        for provider in ["openai", "deepseek", "ollama", "vllm", "llamacpp"]:
            assert provider in out


# ── search ─────────────────────────────────────────────────────────────────────


class TestSearchCommand:
    def test_search_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "--max-results" in out
        assert "--json" in out


# ── ask ───────────────────────────────────────────────────────────────────────


class TestAskCommand:
    def test_ask_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["ask", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "--provider" in out
        assert "--max-iterations" in out
        assert "--fast-mode" in out
        assert "--json" in out


# ── cot ───────────────────────────────────────────────────────────────────────


class TestCotCommand:
    def test_cot_help_shows_modes(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["cot", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "--mode" in out
        assert "--n-samples" in out
        assert "--temperature" in out


# ── tot ───────────────────────────────────────────────────────────────────────


class TestTotCommand:
    def test_tot_help_shows_strategies(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["tot", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "--strategy" in out
        assert "beam" in out
        assert "--branching-factor" in out


# ── serve ─────────────────────────────────────────────────────────────────────


class TestServeCommand:
    def test_serve_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        out = _strip_ansi(result.stdout)
        assert "--host" in out
        assert "--port" in out
        assert "--workers" in out
        assert "--reload" in out


# ── Top-level help ────────────────────────────────────────────────────────────


class TestTopLevelHelp:
    def test_no_args_prints_help(self, runner: CliRunner) -> None:
        result = runner.invoke(app, [])
        # Typer's no_args_is_help=True should exit with code 0 (or 2)
        # and print the help text.
        out = _strip_ansi(result.stdout)
        assert "Usage" in out
        # List all commands
        for cmd in ["version", "ask", "cot", "tot", "search", "serve", "providers"]:
            assert cmd in out
