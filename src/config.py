"""
Configuration module for the document conversion pipeline. Loads application settings
from environment variables or .env file using Pydantic settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    DOCLING_HOST: str
    DOCLING_PORT: int
    DOCLING_TIMEOUT: float
    INPUT_DIR: str
    OUTPUT_DIR: str

    @property
    def docling_url(self) -> str:
        """Get the full Docling API URL."""
        return f"http://{self.DOCLING_HOST}:{self.DOCLING_PORT}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
print(settings.docling_url)
print(settings.DOCLING_TIMEOUT)
