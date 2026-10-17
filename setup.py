"""
Legacy setup.py shim — kept only for backward compatibility with
installers that don't yet support PEP 517 builds (pyproject.toml).

All real metadata now lives in pyproject.toml. This file simply
delegates to setuptools so `pip install .` continues to work.

Author: Adil Shamim <adilshamim8@users.noreply.github.com>
Date:   17 October 2026
"""

from setuptools import setup

setup()
