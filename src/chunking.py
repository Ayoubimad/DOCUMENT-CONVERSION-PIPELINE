"""
Document chunking module that provides strategies for splitting documents into smaller chunks.
Includes an LLM-based chunking strategy that uses AI to find natural breakpoints.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import re
from document import Document

from langchain_text_splitters import (
    CharacterTextSplitter,
    TokenTextSplitter,
)

from llm import OpenAIModel


class ChunkingStrategy(ABC):
    """Base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> List[Document]:
        """Split a document into chunks according to the strategy.

        Args:
            document: The document to split into chunks

        Returns:
            A list of Document objects representing the chunks
        """
        raise NotImplementedError("Subclasses must implement this method")

    def clean_text(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: The text to clean

        Returns:
            Text cleaned of non-ascii characters, base64 images, and normalized whitespace
        """
        # Remove base64 images
        cleaned_text = re.sub(r"!\[.*?\]\(data:image/[^;]*;base64,[^)]*\)", "", text)
        # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        # Replace multiple tabs with a single tab
        cleaned_text = re.sub(r"\t+", "\t", cleaned_text)
        # Replace multiple carriage returns with a single carriage return
        cleaned_text = re.sub(r"\r+", "\r", cleaned_text)
        # Replace multiple form feeds with a single form feed
        cleaned_text = re.sub(r"\f+", "\f", cleaned_text)
        # Replace multiple vertical tabs with a single vertical tab
        cleaned_text = re.sub(r"\v+", "\v", cleaned_text)
        # Remove non ascii characters
        cleaned_text = re.sub(r"[^\x00-\x7F]+", "", cleaned_text)
        return cleaned_text


class LangChainChunking(ChunkingStrategy):
    """Chunking strategy that uses LangChain text splitters."""

    def __init__(
        self,
        splitter_type: str = "tiktoken",
        chunk_size: int = 8000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize the LangChain chunking strategy.

        Args:
            splitter_type: Type of text splitter to use ('character', 'token')
            chunk_size: Target size of each chunk in characters (or tokens for TokenTextSplitter)
            chunk_overlap: Number of characters/tokens to overlap between chunks
            separators: List of separators to use for splitting (for RecursiveCharacterTextSplitter)
            **kwargs: Additional keyword arguments to pass to the text splitter

        Raises:
            ImportError: If LangChain is not installed
            ValueError: If an invalid splitter_type is provided
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        if splitter_type == "character":
            self.splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )
        elif splitter_type == "token":
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid splitter_type: {splitter_type}. Must be one of: "
                "character, token"
            )

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using LangChain text splitters.

        Args:
            document: The document to split into chunks

        Returns:
            List of Document objects representing the chunks
        """
        if len(document.content) <= self.chunk_size:
            return [document]

        clean_content = self.clean_text(document.content)

        text_chunks = self.splitter.split_text(clean_content)

        chunks: List[Document] = []
        for i, chunk_content in enumerate(text_chunks, 1):
            meta_data = document.meta_data.copy()
            meta_data["chunk"] = i
            meta_data["chunk_size"] = len(chunk_content)
            meta_data["splitter_type"] = self.splitter.__class__.__name__

            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{i}"
            elif document.name:
                chunk_id = f"{document.name}_{i}"

            chunks.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk_content,
                )
            )

        return chunks


class AgenticChunking(ChunkingStrategy):
    """Chunking strategy that uses an LLM to determine natural breakpoints in the text"""

    def __init__(self, model: OpenAIModel, max_chunk_size: int = 32000):
        self.max_chunk_size = max_chunk_size
        self.model = model

    def _prepare_chunk(
        self, document: Document, chunk_text: str, chunk_number: int
    ) -> Document:
        """Create a Document object for a chunk of text"""
        meta_data = document.meta_data.copy() if document.meta_data else {}
        meta_data["chunk"] = chunk_number
        meta_data["chunk_size"] = len(chunk_text)
        meta_data["splitter_type"] = "agentic"

        chunk_id = None
        if document.id:
            chunk_id = f"{document.id}_{chunk_number}"
        elif document.name:
            chunk_id = f"{document.name}_{chunk_number}"

        return Document(
            id=chunk_id,
            name=document.name,
            meta_data=meta_data,
            content=chunk_text,
        )

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using LLM to determine natural breakpoints based on context"""
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_number = 1

        while remaining_text:
            prompt = self._create_breakpoint_prompt(remaining_text)

            try:
                break_point = int(self.model.generate(prompt).strip())
            except Exception:
                break_point = self.max_chunk_size

            chunk_text = remaining_text[:break_point].strip()
            chunks.append(self._prepare_chunk(document, chunk_text, chunk_number))
            chunk_number += 1
            remaining_text = remaining_text[break_point:].strip()

            if not remaining_text:
                break

        return chunks

    async def chunk_async(self, document: Document) -> List[Document]:
        """Split text into chunks asynchronously using LLM to determine natural breakpoints"""
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_number = 1

        while remaining_text:
            prompt = self._create_breakpoint_prompt(remaining_text)

            try:
                break_point = int((await self.model.generate_async(prompt)).strip())
            except Exception:
                break_point = self.max_chunk_size

            chunk_text = remaining_text[:break_point].strip()
            chunks.append(self._prepare_chunk(document, chunk_text, chunk_number))
            chunk_number += 1
            remaining_text = remaining_text[break_point:].strip()

            if not remaining_text:
                break

        return chunks

    def _create_breakpoint_prompt(self, text: str) -> str:
        """Create a prompt for finding a natural breakpoint in text"""
        return f"""Analyze this text and determine a natural breakpoint within the first {self.max_chunk_size} characters.
        Consider semantic completeness, paragraph boundaries, and topic transitions.
        **NEVER** break words, sentences or paragraphs.
        **NEVER** break tables or lists.
        **NEVER** break code blocks.
        **NEVER** break links.
        Return only the character position number of where to break the text:

        {text[:self.max_chunk_size]}"""
