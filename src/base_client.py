"""
Base HTTP client module providing common functionality for API communication.
Implements async HTTP request handling, error management, context manager support,
and resource cleanup for clients that interact with document conversion APIs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import httpx
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseHttpClient(ABC):
    """Base class for HTTP clients with common functionality.

    Implements common HTTP client functionality including connection management,
    request handling, and resource cleanup.
    """

    def __init__(self, *, base_url: str, timeout: float = 60.0):
        """Initialize the HTTP client.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=self.timeout)
        logger.debug(
            f"Initialized HTTP client with base URL: {base_url}, default timeout: {self.timeout}s"
        )

    async def close(self) -> None:
        """Close the HTTP client connection and release resources."""
        if hasattr(self, "_client") and self._client:
            await self._client.aclose()

    async def __aenter__(self) -> "BaseHttpClient":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and ensure resources are released."""
        await self.close()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            timeout: Optional specific timeout for this request (overrides default)
            **kwargs: Additional arguments for the request

        Returns:
            Dict[str, Any]: Response data

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response is not valid JSON
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:

            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            logger.error(f"Request timeout after {self.timeout}s for {method} {url}")
            raise
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_msg = f"HTTP Error {status_code} for {method} {url}"

            try:
                error_detail = e.response.json()
                error_msg += f": {error_detail}"
            except (ValueError, KeyError):
                if e.response.text:
                    error_msg += f": {e.response.text[:200]}"

            raise httpx.HTTPStatusError(
                error_msg, request=e.request, response=e.response
            ) from e
        except httpx.RequestError as e:
            raise httpx.RequestError(
                f"Request failed for {method} {url}: {str(e)}", request=e.request
            ) from e

    @abstractmethod
    async def convert_file(
        self,
        file_path: Path,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a file using the API.

        Args:
            file_path: Path to the file to convert
            options: Conversion options
            timeout: Optional specific timeout for this conversion

        Returns:
            Dict[str, Any]: Conversion result
        """
        raise NotImplementedError("Subclasses must implement convert_file()")
