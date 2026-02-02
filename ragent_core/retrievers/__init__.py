from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    doc_id: Union[int, str]
    score: float
    text: str
    document: Optional[dict[str, Any]] = None


from .base import BaseRetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever


__all__ = [
    "HybridRetriever",
    "BM25Retriever",
    "RetrievalResult",
    "BaseRetriever",
]
