"""Externalized runtime configuration and structured logging helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_SLANG_URL = (
    "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/"
    "colloquial-indonesian-lexicon.csv"
)


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    """Parse boolean environment strings consistently."""
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_cors_origins(raw_value: str | None) -> tuple[str, ...]:
    """Parse comma-separated CORS origins from environment configuration."""
    if not raw_value or not raw_value.strip():
        return ("http://localhost:3000", "http://localhost:8000")
    origins = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    return origins or ("http://localhost:3000",)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from environment variables."""

    app_name: str = "youtube-sentiment-svm"
    app_env: str = "development"
    log_level: str = "INFO"
    model_path: Path = Path("artifacts/sentiment_pipeline.joblib")
    allow_remote_lexicon: bool = False
    slang_dict_url: str = DEFAULT_SLANG_URL
    http_timeout_seconds: float = 5.0
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_per_minute: int = 120
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:8000")

    def __post_init__(self) -> None:
        """Validate numeric configuration boundaries upfront."""
        if self.http_timeout_seconds <= 0:
            raise ValueError("HTTP_TIMEOUT_SECONDS must be positive.")
        if self.rate_limit_per_minute <= 0:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be positive.")
        if not 1 <= self.api_port <= 65535:
            raise ValueError("API_PORT must be between 1 and 65535.")


def configure_logging(level: str | None = None) -> None:
    """Configure root structured leveled logging once per process.

    Args:
        level: Optional logging level override (e.g., ``INFO``, ``DEBUG``).
    """
    raw_level = (
        level if (level is not None and level.strip()) else (os.getenv("LOG_LEVEL") or "INFO")
    )
    resolved_level_name = raw_level.strip().upper()
    resolved_level = getattr(logging, resolved_level_name, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(resolved_level)
    logger.info("Configured structured logging at level %s", resolved_level_name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache runtime settings from ``.env`` and environment variables.

    Returns:
        Validated ``Settings`` instance.
    """
    load_dotenv(override=False)
    timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "5.0"))
    rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    port = int(os.getenv("API_PORT", "8000"))

    return Settings(
        app_name=os.getenv("APP_NAME", "youtube-sentiment-svm"),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        model_path=Path(os.getenv("MODEL_PATH", "artifacts/sentiment_pipeline.joblib")),
        allow_remote_lexicon=_parse_bool(os.getenv("ALLOW_REMOTE_LEXICON"), default=False),
        slang_dict_url=os.getenv("SLANG_DICT_URL", DEFAULT_SLANG_URL),
        http_timeout_seconds=timeout,
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=port,
        rate_limit_per_minute=rate_limit,
        cors_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS")),
    )


def clear_settings_cache() -> None:
    """Clear the cached settings singleton (primarily for isolated testing)."""
    get_settings.cache_clear()
