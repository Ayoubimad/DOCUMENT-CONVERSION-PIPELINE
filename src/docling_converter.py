"""
Docling converter implementation that converts documents using the Docling API.
Provides both synchronous and asynchronous methods for document conversion, with error
handling and proper resource management for the underlying API client.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from docling.datamodel.base_models import OutputFormat
from base_converter import DocumentConverter
from document import Document
from conversion_option import ConvertDocumentsOptions
from docling_client import DoclingClient

logger = logging.getLogger(__name__)


class DoclingConverter(DocumentConverter):
    """Converter that uses the docling API to convert documents."""

    def __init__(self, options: Optional[ConvertDocumentsOptions] = None):
        """Initialize the docling converter.

        Args:
            options: Optional conversion options
        """
        super().__init__()
        self.client = DoclingClient()
        self.options = options or ConvertDocumentsOptions()
        if OutputFormat.MARKDOWN not in self.options.to_formats:
            self.options.to_formats.append(OutputFormat.MARKDOWN)

    def convert(self, input_source: str, **kwargs: Any) -> Document:
        """Convert a single document synchronously.

        Args:
            input_source: Path to the input document
            **kwargs: Additional conversion options

        Returns:
            Document: The converted document
        """
        return asyncio.run(self.convert_async(input_source, **kwargs))

    def convert_all(self, input_sources: List[str], **kwargs: Any) -> List[Document]:
        """Convert multiple documents synchronously.

        Args:
            input_sources: List of paths to input documents
            **kwargs: Additional conversion options

        Returns:
            List[Document]: List of converted documents
        """
        return asyncio.run(self.convert_all_async(input_sources, **kwargs))

    async def convert_async(self, input_source: str, **kwargs: Any) -> Document:
        """Convert a single document asynchronously.

        Args:
            input_source: Path to the input document
            **kwargs: Additional conversion options

        Returns:
            Document: The converted document
        """
        logger.info(f"Converting document: {input_source}")
        file_path = Path(input_source)

        try:
            result = await self.client.convert_file(
                file_path,
                options=self.options,
                timeout=self.client.timeout,
            )

            return self._create_document(result)

        except asyncio.TimeoutError:
            logger.error(f"Timeout calling Docling API for {input_source}")
            raise
        except Exception as e:
            logger.error(f"Error calling Docling API: {e}", exc_info=True)
            raise

    async def convert_all_async(
        self, input_sources: List[str], **kwargs: Any
    ) -> List[Document]:
        """Convert multiple documents asynchronously.

        Args:
            input_sources: List of paths to input documents
            **kwargs: Additional conversion options

        Returns:
            List[Document]: List of converted documents
        """

        async def safe_convert(input_source: str) -> Optional[Document]:
            """Convert a document safely, catching exceptions.

            Args:
                input_source: Path to the input document

            Returns:
                Optional[Document]: The converted document or None if conversion failed
            """
            try:
                return await self.convert_async(input_source, **kwargs)
            except Exception as e:
                logger.error(
                    f"Failed to convert {input_source}: {str(e)}", exc_info=True
                )
                return None

        tasks = [safe_convert(input_source) for input_source in input_sources]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc is not None]

    async def close(self) -> None:
        """Close the client connection."""
        await self.client.close()

    @staticmethod
    def _create_document(response: Dict[str, Any]) -> Document:
        """Create a Document from the API response.

        Args:
            response: API response dictionary

        Returns:
            Document: The created document
        """
        document = response.get("document", {})
        content = document.get("md_content", "")
        meta_data = document.get("meta_data", {})

        return Document(
            content=content,
            id=document.get("id"),
            name=document.get("name"),
            meta_data=meta_data or {},
        )
