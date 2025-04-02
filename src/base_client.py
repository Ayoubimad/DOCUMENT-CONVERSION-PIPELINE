from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import httpx
from pathlib import Path


class BaseHttpClient(ABC):
    """Base class for HTTP clients with common functionality."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        """Initialize the HTTP client.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self._client.aclose()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for the request

        Returns:
            Dict[str, Any]: Response data

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response is not valid JSON
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    @abstractmethod
    async def convert_file(
        self,
        file_path: Path,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert a file using the API.

        Args:
            file_path: Path to the file to convert
            options: Conversion options

        Returns:
            Dict[str, Any]: Conversion result
        """
        raise NotImplementedError("Subclasses must implement convert_file()")
