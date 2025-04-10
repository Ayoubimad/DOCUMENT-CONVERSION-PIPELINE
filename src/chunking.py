"""
Document chunking module that provides strategies for splitting documents into smaller chunks.
Includes an LLM-based chunking strategy that uses AI to find natural breakpoints.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import re
from document import Document

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter,
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
        pass

    def clean_text(self, text: str) -> str:
        """Clean text by removing base64 images from markdown.

        Args:
            text: The text to clean

        Returns:
            Cleaned text with normalized whitespace and no base64 images
        """
        text = re.sub(r"!\[.*?\]\(data:image/[^;]*;base64,[^)]*\)", "", text)
        return text


class LangChainChunking(ChunkingStrategy):
    """Chunking strategy that uses LangChain text splitters."""

    def __init__(
        self,
        splitter_type: str = "recursive",
        chunk_size: int = 5000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize the LangChain chunking strategy.

        Args:
            splitter_type: Type of text splitter to use ('recursive', 'character', 'markdown', 'token')
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

        if splitter_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                **kwargs,
            )
        elif splitter_type == "character":
            self.splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separators[0] if separators else "\n\n",
                **kwargs,
            )
        elif splitter_type == "markdown":
            self.splitter = MarkdownTextSplitter(
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
                "recursive, character, markdown, token"
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
    """Chunking strategy that uses an LLM to determine natural breakpoints in the text."""

    def __init__(self, model: Optional[OpenAIModel] = None, max_chunk_size: int = 5000):
        """Initialize the agentic chunking strategy.

        Args:
            model: The language model to use for finding breakpoints
            max_chunk_size: Maximum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
        self.model = model
        if model is None:
            raise ValueError("A language model must be provided for AgenticChunking")

    def chunk(self, document: Document) -> List[Document]:
        """Split text into chunks using LLM to determine natural breakpoints based on context.

        Args:
            document: The document to split into chunks

        Returns:
            List of Document objects representing the chunks
        """
        if len(document.content) <= self.max_chunk_size:
            return [document]

        chunks: List[Document] = []
        remaining_text = self.clean_text(document.content)
        chunk_meta_data = document.meta_data
        chunk_number = 1

        while remaining_text:
            prompt = f"""Analyze this text and determine a natural breakpoint within the first {self.max_chunk_size} characters.
            Consider semantic completeness, paragraph boundaries, and topic transitions.
            Return only the character position number of where to break the text:

            {remaining_text[:self.max_chunk_size]}"""

            try:
                break_point = min(
                    int(self.model.generate(prompt).strip()), self.max_chunk_size
                )
                print(f"Break point: {break_point}")
            except Exception:
                break_point = self.max_chunk_size

            chunk = remaining_text[:break_point].strip()
            meta_data = chunk_meta_data.copy()
            meta_data["chunk"] = chunk_number
            meta_data["chunk_size"] = len(chunk)

            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{chunk_number}"
            elif document.name:
                chunk_id = f"{document.name}_{chunk_number}"

            chunks.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk,
                )
            )
            chunk_number += 1

            remaining_text = remaining_text[break_point:].strip()

            if not remaining_text:
                break

        return chunks
