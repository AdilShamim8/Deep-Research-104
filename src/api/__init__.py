"""
FastAPI application package for Deep Research 104.

Public surface:
    - app       — the configured FastAPI instance (uvicorn target)
    - router    — the API router (mountable elsewhere)
    - schemas   — request/response Pydantic models

Date tag: 17 October 2026.
"""

from src.api.main import app, create_app
from src.api.routes import router

__all__ = ["app", "create_app", "router"]
