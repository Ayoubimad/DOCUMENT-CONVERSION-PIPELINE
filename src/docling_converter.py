"""
Docling converter implementation that converts documents using the Docling API.
Provides both synchronous and asynchronous methods for document conversion, with error
handling and proper resource management for the underlying API client.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time

from docling.datamodel.base_models import OutputFormat
from base_converter import DocumentConverter
from document import Document
from conversion_option import ConvertDocumentsOptions
from docling_client import DoclingClient
from config import settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
MAX_RETRY_DELAY = 60


class DoclingConverter(DocumentConverter):
    """Converter that uses the docling API to convert documents."""

    def __init__(self, options: Optional[ConvertDocumentsOptions] = None):
        """Initialize the docling converter.

        Args:
            options: Optional conversion options
        """
        super().__init__()
        self.client = DoclingClient(
            base_url=settings.docling_url, timeout=settings.DOCLING_TIMEOUT
        )
        self.options = options or ConvertDocumentsOptions()

        logger.info(
            f"Initializing DoclingConverter with URL: {self.client.base_url}, timeout: {self.client.timeout}"
        )

        # Ensure markdown is in the output formats
        if OutputFormat.MARKDOWN not in self.options.to_formats:
            self.options.to_formats.append(OutputFormat.MARKDOWN)

    async def convert_async(self, input_source: str, **kwargs: Any) -> Document:
        """Convert a single document asynchronously with retry mechanism.

        Args:
            input_source: Path to the input document
            **kwargs: Additional conversion options

        Returns:
            Document: The converted document

        Raises:
            Exception: If the document could not be converted after retries
        """
        logger.info(f"Converting document: {input_source}")
        file_path = Path(input_source)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_source}")

        max_retries = kwargs.get("max_retries", MAX_RETRIES)
        retry_delay = kwargs.get("retry_delay", RETRY_DELAY_BASE)

        last_exception = None

        for attempt in range(max_retries):
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                timeout_multiplier = max(1.0, min(10.0, file_size_mb / 10))
                timeout = self.client.timeout * timeout_multiplier

                logger.debug(
                    f"Attempt {attempt+1}/{max_retries} for {input_source} (size: {file_size_mb:.2f}MB, timeout: {timeout:.1f}s)"
                )

                result = await self.client.convert_file(
                    file_path,
                    options=self.options,
                    timeout=timeout,
                )

                return self._create_document(result)

            except asyncio.TimeoutError as e:
                logger.warning(
                    f"Timeout on attempt {attempt+1}/{max_retries} for {input_source}"
                )
                last_exception = e
            except Exception as e:
                logger.warning(
                    f"Error on attempt {attempt+1}/{max_retries} for {input_source}: {e}"
                )
                last_exception = e

            if attempt < max_retries - 1:
                jitter = 0.8 + 0.4 * (hash(input_source) % 10) / 10
                current_delay = min(
                    MAX_RETRY_DELAY, retry_delay * (2**attempt) * jitter
                )
                logger.info(
                    f"Retrying {input_source} in {current_delay:.1f} seconds..."
                )
                await asyncio.sleep(current_delay)

        error_msg = f"Failed to convert {input_source} after {max_retries} attempts"
        logger.error(error_msg)
        if last_exception:
            raise type(last_exception)(f"{error_msg}: {last_exception}")
        else:
            raise RuntimeError(error_msg)

    async def convert_all_async(
        self, input_sources: List[str], **kwargs: Any
    ) -> List[Document]:
        """Convert multiple documents asynchronously with concurrency control.

        Args:
            input_sources: List of paths to input documents
            **kwargs: Additional conversion options including:
                      - max_concurrent: Maximum number of concurrent conversions
                      - max_retries: Maximum number of retries per document
                      - retry_delay: Base delay for retry mechanism

        Returns:
            List[Document]: List of converted documents
        """
        if not input_sources:
            return []

        max_concurrent = kwargs.get("max_concurrent", min(5, os.cpu_count() or 1))
        logger.info(
            f"Converting {len(input_sources)} documents with max concurrency of {max_concurrent}"
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def safe_convert(input_source: str) -> Optional[Document]:
            """Convert a document safely with concurrency control.

            Args:
                input_source: Path to the input document

            Returns:
                Optional[Document]: The converted document or None if conversion failed
            """
            async with semaphore:
                try:
                    return await self.convert_async(input_source, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to convert {input_source}: {str(e)}")
                    return None

        tasks = [safe_convert(input_source) for input_source in input_sources]

        results = await asyncio.gather(*tasks)

        return [doc for doc in results if doc is not None]

    def convert(self, input_source: str, **kwargs: Any) -> Document:
        """Convert a single document synchronously by running the async version.

        Args:
            input_source: Path to the input document
            **kwargs: Additional conversion options

        Returns:
            Document: The converted document
        """
        return asyncio.run(self.convert_async(input_source, **kwargs))

    def convert_all(self, input_sources: List[str], **kwargs: Any) -> List[Document]:
        """Convert multiple documents synchronously by running the async version.

        Args:
            input_sources: List of paths to input documents
            **kwargs: Additional conversion options

        Returns:
            List[Document]: List of converted documents
        """
        return asyncio.run(self.convert_all_async(input_sources, **kwargs))

    async def close(self) -> None:
        """Close the client connection."""
        try:
            await self.client.close()
            logger.debug("Closed DoclingClient connection")
        except Exception as e:
            logger.error(f"Error closing DoclingClient: {e}")

    @staticmethod
    def _create_document(response: Dict[str, Any]) -> Document:
        """Create a Document from the API response.

        Args:
            response: API response dictionary

        Returns:
            Document: The created document

        Raises:
            ValueError: If the response doesn't contain required document data
        """
        if not response:
            raise ValueError("Empty response from Docling API")

        document = response.get("document", {})
        if not document:
            raise ValueError(f"No document data in Docling API response: {response}")

        content = document.get("md_content", "")
        if not content:
            logger.warning(
                f"Document has empty content: {document.get('id', 'unknown')}"
            )

        meta_data = document.get("meta_data", {})

        return Document(
            content=content,
            id=document.get("id"),
            name=document.get("name"),
            meta_data=meta_data or {},
        )
