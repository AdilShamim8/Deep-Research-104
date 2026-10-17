"""
Allow `python -m src.cli.app ...` as a Typer CLI invocation.

Equivalent to `deep-research ...` once the package is installed via
pip install -e .
"""

from src.cli.app import main


if __name__ == "__main__":
    main()
