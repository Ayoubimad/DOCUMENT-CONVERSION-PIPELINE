from typing import Any, Dict, Optional

from pydantic import BaseModel


class Document(BaseModel):
    """Dataclass for managing a document"""

    content: str
    id: Optional[str] = None
    name: Optional[str] = None
    meta_data: Dict[str, Any] = {}
