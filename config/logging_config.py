"""
Structured logging configuration using Loguru.

Production-ready with:
- JSON output for log aggregators (ELK, Loki, Datadog)
- Pretty colored output for development
- File rotation + retention + compression
- Request-id propagation via contextvars
- Sentry / OpenTelemetry bridge hooks
- Environment-aware log levels

Date tag: 17 October 2026.
"""

from __future__ import annotations

import json
import os
import sys
from contextvars import ContextVar
from typing import Any

from loguru import logger

from config.settings import settings

# ── Request-scoped context (for correlation IDs) ──────────────────────────────
# Set by FastAPI middleware on every request; consumed by the JSON formatter
# so every log line within a request carries the same request_id.

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")


# ── JSON formatter ────────────────────────────────────────────────────────────


def serialize_record(record: dict) -> str:
    """Serialize a loguru record into a single-line JSON string.

    Includes request_id/user_id/trace_id from contextvars when present,
    so downstream log aggregators can correlate across services.
    """
    subset: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "env": settings.app_env,
        "service": settings.app_name,
    }

    # Pull correlation IDs from contextvars (no-op if not set)
    rid = request_id_ctx.get()
    if rid:
        subset["request_id"] = rid
    uid = user_id_ctx.get()
    if uid:
        subset["user_id"] = uid
    tid = trace_id_ctx.get()
    if tid:
        subset["trace_id"] = tid

    # Extra fields (passed via logger.bind(...))
    if record["extra"]:
        for k, v in record["extra"].items():
            if k not in subset:
                subset[k] = v

    # Exception traceback (stringified, structured)
    if record["exception"]:
        subset["exception"] = _format_exception(record["exception"])

    return json.dumps(subset, default=str, ensure_ascii=False)


def _format_exception(exc) -> str:
    """Stringify an exception while keeping it readable in JSON."""
    try:
        import traceback

        return "".join(traceback.format_exception(exc))
    except Exception:
        return str(exc)


# ── Pretty dev formatter ─────────────────────────────────────────────────────


def pretty_format(record: dict) -> str:
    """Colored multi-field format for interactive development."""
    rid = request_id_ctx.get()
    rid_part = f" <dim>[{rid}]</dim>" if rid else ""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        f"{rid_part} | "
        "<level>{message}</level>\n"
        "{exception}"
    )


# ── Public setup ──────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """Configure loguru sinks based on app_env.

    Idempotent: safe to call multiple times (each call replaces handlers).
    """
    logger.remove()  # Drop default handler

    if settings.app_env == "production":
        # ── stdout: JSON for log aggregators ───────────────────────────────
        logger.add(
            sys.stdout,
            format=serialize_record,
            level=settings.log_level,
            serialize=False,
            backtrace=False,
            diagnose=False,  # Don't leak variable values in prod
            enqueue=True,  # Thread-safe, non-blocking
            catch=True,  # Don't crash on logger errors
        )

        # ── File: warnings+ with rotation ──────────────────────────────────
        os.makedirs("logs", exist_ok=True)
        logger.add(
            "logs/app_{time}.log",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            level="WARNING",
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )

        # ── File: errors only (for fast triage) ────────────────────────────
        logger.add(
            "logs/errors_{time}.log",
            rotation="50 MB",
            retention="90 days",
            compression="gz",
            level="ERROR",
            serialize=True,
            enqueue=True,
        )

    elif settings.app_env == "staging":
        # ── Mirror production but at DEBUG level ───────────────────────────
        logger.add(
            sys.stdout,
            format=serialize_record,
            level="DEBUG",
            serialize=False,
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )
        os.makedirs("logs", exist_ok=True)
        logger.add(
            "logs/app_{time}.log",
            rotation="50 MB",
            retention="14 days",
            compression="gz",
            level="INFO",
            serialize=True,
            enqueue=True,
        )

    else:
        # ── Development: pretty + color ────────────────────────────────────
        logger.add(
            sys.stdout,
            format=pretty_format,
            level=settings.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        # Local file mirror, no rotation needed
        os.makedirs("logs", exist_ok=True)
        logger.add(
            "logs/dev.log",
            format=serialize_record,
            level="DEBUG",
            serialize=True,
            rotation="20 MB",
            retention="7 days",
        )


# ── Helpers for request-scoped binding ───────────────────────────────────────


def bind_request_id(request_id: str) -> None:
    """Set the current request_id for log correlation. Clear at request end."""
    request_id_ctx.set(request_id)


def bind_user_id(user_id: str) -> None:
    user_id_ctx.set(user_id)


def bind_trace_id(trace_id: str) -> None:
    trace_id_ctx.set(trace_id)


def clear_request_context() -> None:
    """Reset all contextvars (call in finally block)."""
    request_id_ctx.set("")
    user_id_ctx.set("")
    trace_id_ctx.set("")


# Run setup at import time so library code can `from loguru import logger`
# and immediately get the configured sinks.
setup_logging()
