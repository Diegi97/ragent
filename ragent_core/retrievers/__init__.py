from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    doc_id: int
    chunk_id: int
    score: float
    text: str


from .base import BaseRetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever


__all__ = [
    "HybridRetriever",
    "BM25Retriever",
    "RetrievalResult",
    "BaseRetriever",
]
