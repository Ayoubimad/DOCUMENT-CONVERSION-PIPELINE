import asyncio
from typing import List, Optional

from docling.datamodel.base_models import OutputFormat
from base_converter import DocumentConverter
from document import Document
from convertion_option import ConvertDocumentsOptions
from docling_client import DoclingClient
import logging

logger = logging.getLogger(__name__)


class DoclingConverter(DocumentConverter):
    """Converter that uses the docling API to convert documents."""

    def __init__(self, options: Optional[ConvertDocumentsOptions] = None):
        """Initialize the docling converter.

        Args:
            options: Optional conversion options
        """
        self.client = DoclingClient()
        self.options = options or ConvertDocumentsOptions()
        # Ensure the output format is at least markdown
        if OutputFormat.MARKDOWN not in self.options.to_formats:
            self.options.to_formats.append(OutputFormat.MARKDOWN)

    def convert(self, input_source: str, **kwargs) -> Document:
        """Convert a single document synchronously.

        Args:
            input_source: Path to the input document

        Returns:
            Document: The converted document
        """
        response = self.client.convert_file(input_source, options=self.options)
        return self._create_document(response)

    def convert_all(self, input_sources: List[str], **kwargs) -> List[Document]:
        """Convert multiple documents synchronously.

        Args:
            input_sources: List of paths to input documents

        Returns:
            List[Document]: List of converted documents
        """
        return asyncio.run(self.convert_all_async(input_sources))

    async def convert_async(self, input_source: str, **kwargs) -> Document:
        """Convert a single document asynchronously.

        Args:
            input_source: Path to the input document

        Returns:
            Document: The converted document
        """
        response = await self.client.convert_file(input_source, options=self.options)
        return self._create_document(response)

    async def convert_all_async(
        self, input_sources: List[str], **kwargs
    ) -> List[Document]:
        """Convert multiple documents asynchronously.

        Args:
            input_sources: List of paths to input documents

        Returns:
            List[Document]: List of converted documents
        """

        # this may produce an error if all documents in output are None

        async def safe_convert(input_source: str) -> Optional[Document]:
            try:
                return await self.convert_async(input_source)
            except Exception as e:
                logger.error(f"Failed to convert {input_source}: {str(e)}")
                return None

        tasks = [safe_convert(input_source) for input_source in input_sources]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc is not None]

    @staticmethod
    def _create_document(response: dict) -> Document:
        """Create a Document from the API response.

        Args:
            response: API response dictionary

        Returns:
            Document: The created document
        """
        return Document(content=response.get("document", {}).get("md_content", ""))
