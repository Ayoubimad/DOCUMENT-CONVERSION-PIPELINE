"""
Configuration module for the document conversion pipeline. Loads application settings
from environment variables or .env file using Pydantic settings.
"""

import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from logging_utils import LoggingConfig, configure_root_logger


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Docling API settings
    DOCLING_HOST: str
    DOCLING_PORT: int
    DOCLING_TIMEOUT: float = 300.0  # Default timeout in seconds
    DOCLING_SSL: bool = False  # Use HTTPS

    # Directory settings
    INPUT_DIR: str
    OUTPUT_DIR: str

    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0  # Base delay for exponential backoff
    MAX_RETRY_DELAY: float = 60.0  # Maximum retry delay

    # Performance settings
    MAX_CONCURRENT_CONVERSIONS: int = 3
    STABILITY_CHECKS: int = 3  # Number of checks before a file is considered stable
    STABILITY_INTERVAL: float = 0.5  # Interval between stability checks

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Optional[str] = None
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: Optional[str] = None
    INCLUDE_TIMESTAMPS: bool = True

    @property
    def docling_url(self) -> str:
        """Get the full Docling API URL with protocol."""
        protocol = "https" if self.DOCLING_SSL else "http"
        return f"{protocol}://{self.DOCLING_HOST}:{self.DOCLING_PORT}"

    def configure_logging(self) -> None:
        """Configure the application logging based on settings."""
        logging_config = LoggingConfig(
            level=self.LOG_LEVEL,
            include_timestamps=self.INCLUDE_TIMESTAMPS,
            log_to_file=self.LOG_TO_FILE,
            log_file_path=self.LOG_FILE_PATH,
            log_format=self.LOG_FORMAT,
        )

        configure_root_logger(logging_config)
        logger = logging.getLogger(__name__)
        logger.debug("Logging configured")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


# Initialize settings
settings = Settings()

# Create required directories
os.makedirs(settings.INPUT_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

# Configure logging
settings.configure_logging()

logger = logging.getLogger(__name__)
logger.debug(
    f"Loaded configuration: DOCLING_URL={settings.docling_url}, "
    f"TIMEOUT={settings.DOCLING_TIMEOUT}, INPUT_DIR={settings.INPUT_DIR}, "
    f"OUTPUT_DIR={settings.OUTPUT_DIR}"
)
