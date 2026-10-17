"""
Configuration package for Deep Research 104.

Public surface:
- `settings`: master AppSettings singleton (env-driven, validated).
- `ModelConfig`, `SearchConfig`, `ReasoningConfig`, `SecurityConfig`,
  `ObservabilityConfig`, `StorageConfig`: typed nested config groups.

Importing this package is cheap: it triggers env validation exactly
once per process thanks to the lru_cache in settings.py.

Date tag: 17 October 2026.
"""

from config.settings import (
    AppSettings,
    ModelConfig,
    ObservabilityConfig,
    ReasoningConfig,
    SearchConfig,
    SecurityConfig,
    StorageConfig,
    settings,
)

__all__ = [
    "AppSettings",
    "ModelConfig",
    "ObservabilityConfig",
    "ReasoningConfig",
    "SearchConfig",
    "SecurityConfig",
    "StorageConfig",
    "settings",
]

__version__ = "1.0.0"
__build_date__ = "2026-10-17"
