"""
CLI package for Deep Research 104.

Public surface:
    - main() — Typer entry point (referenced by [project.scripts] in
      pyproject.toml as `deep-research`)

Date tag: 17 October 2026.
"""

from src.cli.app import main

__all__ = ["main"]
