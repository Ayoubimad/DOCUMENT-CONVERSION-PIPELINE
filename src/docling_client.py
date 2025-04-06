"""
Docling API client module that provides specific implementation for interacting with
the Docling document conversion service. Supports converting documents via URLs, local files,
and base64-encoded content through the Docling HTTP API.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from base_client import BaseHttpClient
from conversion_option import ConvertDocumentsOptions
from config import settings

logger = logging.getLogger(__name__)


class DoclingClient(BaseHttpClient):
    """Client for interacting with the docling-server API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize the docling client.

        Args:
            base_url: Optional base URL of the docling server (defaults to env setting)
            timeout: Optional request timeout in seconds (defaults to env setting)
        """
        super().__init__(
            base_url=base_url or settings.docling_url,
            timeout=timeout or settings.DOCLING_TIMEOUT,
        )
        logger.debug(
            f"Initialized DoclingClient with URL: {self.base_url}, timeout: {self.timeout}s"
        )

    async def convert_url(
        self,
        url: str,
        options: Optional[ConvertDocumentsOptions] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a document from a URL.

        Args:
            url: URL of the document to convert
            options: Conversion options
            headers: Optional HTTP headers for the request
            timeout: Optional specific timeout for this conversion

        Returns:
            Dict[str, Any]: Conversion result
        """
        payload = {
            "http_sources": [{"url": url, "headers": headers or {}}],
            "options": options.model_dump(exclude_none=True) if options else None,
        }
        logger.debug(f"Converting document from URL: {url}")
        return await self._make_request(
            "POST", "v1alpha/convert/source", json=payload, timeout=timeout
        )

    async def convert_file(
        self,
        file_path: Union[str, Path],
        options: Optional[ConvertDocumentsOptions] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a local file.

        Args:
            file_path: Path to the file to convert
            options: Conversion options
            timeout: Optional specific timeout for this conversion

        Returns:
            Dict[str, Any]: Conversion result

        Raises:
            FileNotFoundError: If the file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Log file size for debugging timeout issues
        file_size = file_path.stat().st_size
        file_size_mb = file_size / 1024 / 1024
        logger.debug(f"Converting file: {file_path.name} ({file_size_mb:.2f} MB)")

        # Adjust timeout for large files if not explicitly specified
        if not timeout and file_size > 5 * 1024 * 1024:  # 5MB
            # For large files, scale the timeout based on file size, if not explicitly set
            scaled_timeout = min(
                self.timeout * (file_size_mb / 5), settings.CONVERSION_MAX_TIMEOUT
            )
            logger.debug(f"Adjusting timeout for large file to {scaled_timeout:.2f}s")
            timeout = scaled_timeout

        files = {
            "files": (file_path.name, open(file_path, "rb"), "application/octet-stream")
        }

        data: Dict[str, Any] = {}
        if options:
            data["parameters"] = json.dumps(options.model_dump(exclude_none=True))

        return await self._make_request(
            "POST", "v1alpha/convert/file", files=files, data=data, timeout=timeout
        )

    async def convert_base64(
        self,
        base64_string: str,
        filename: str,
        options: Optional[ConvertDocumentsOptions] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a base64-encoded file.

        Args:
            base64_string: Base64-encoded file content
            filename: Name of the file
            options: Conversion options
            timeout: Optional specific timeout for this conversion

        Returns:
            Dict[str, Any]: Conversion result
        """
        # Estimate the size of the decoded content for logging
        decoded_size_mb = (
            len(base64_string) * 0.75 / 1024 / 1024
        )  # base64 is ~4/3 the size of binary
        logger.debug(
            f"Converting base64 data for {filename} (approx. {decoded_size_mb:.2f} MB)"
        )

        payload = {
            "file_sources": [{"base64_string": base64_string, "filename": filename}],
            "options": options.model_dump(exclude_none=True) if options else None,
        }
        return await self._make_request(
            "POST", "v1alpha/convert/source", json=payload, timeout=timeout
        )

    @staticmethod
    def encode_file_to_base64(file_path: Union[str, Path]) -> str:
        """Encode a file to base64.

        Args:
            file_path: Path to the file to encode

        Returns:
            str: Base64-encoded string
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
