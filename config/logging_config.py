"""
Structured logging configuration using Loguru.
Production-ready with JSON output and trace IDs.
"""

import sys
import json
from loguru import logger
from config.settings import settings


def serialize_record(record: dict) -> str:
    """Serialize log record to JSON for production."""
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    if record["extra"]:
        subset["extra"] = record["extra"]
    if record["exception"]:
        subset["exception"] = str(record["exception"])
    return json.dumps(subset)


def setup_logging() -> None:
    """Configure logging based on environment."""
    logger.remove()  # Remove default handler
    
    if settings.app_env == "production":
        # JSON structured logging for production
        logger.add(
            sys.stdout,
            format=serialize_record,
            level=settings.log_level,
            serialize=False,
            backtrace=False,
            diagnose=False,
        )
        # Also write to file with rotation
        logger.add(
            "logs/app_{time}.log",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            level="WARNING",
            serialize=True,
        )
    else:
        # Pretty logging for development
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=settings.log_level,
            colorize=True,
        )


setup_logging()