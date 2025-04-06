from abc import ABC, abstractmethod
from typing import Any, List, Optional

from document import Document


class DocumentConverter(ABC):
    """Abstract base class for document converters.

    This class defines the interface that all document converters must implement.
    It provides both synchronous and asynchronous methods for converting documents.
    """

    @abstractmethod
    def convert(
        self,
        input_source: str,
        **kwargs: Any,
    ) -> Document:
        """Convert a single document synchronously.

        Args:
            input_source: Path to the input document file
            **kwargs: Additional conversion options

        Returns:
            Document: The converted document

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement convert()")

    @abstractmethod
    def convert_all(
        self,
        input_sources: List[str],
        **kwargs: Any,
    ) -> List[Document]:
        """Convert multiple documents synchronously.

        Args:
            input_sources: List of paths to input document files
            **kwargs: Additional conversion options

        Returns:
            List[Document]: List of converted documents

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement convert_all()")

    @abstractmethod
    async def convert_async(
        self,
        input_source: str,
        **kwargs: Any,
    ) -> Document:
        """Convert a single document asynchronously.

        Args:
            input_source: Path to the input document file
            **kwargs: Additional conversion options

        Returns:
            Document: The converted document

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement convert_async()")

    @abstractmethod
    async def convert_all_async(
        self,
        input_sources: List[str],
        **kwargs: Any,
    ) -> List[Document]:
        """Convert multiple documents asynchronously.

        Args:
            input_sources: List of paths to input document files
            **kwargs: Additional conversion options

        Returns:
            List[Document]: List of converted documents

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement convert_all_async()")

    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the converter.

        Should be called when the converter is no longer needed to ensure proper
        resource cleanup.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement close()")
