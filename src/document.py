"""
Document model module containing the core document representation used throughout
the conversion pipeline. Provides storage for document content, metadata and identifiers.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Model for representing a document in the conversion pipeline.

    Attributes:
        content: The text content of the document
        id: Unique identifier for the document
        name: User-friendly name of the document
        meta_data: Additional metadata associated with the document
    """

    content: str = Field(description="The text content of the document")
    id: Optional[str] = Field(
        default=None, description="Unique identifier for the document"
    )
    name: Optional[str] = Field(
        default=None, description="User-friendly name of the document"
    )
    meta_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the document",
    )
