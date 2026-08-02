from enum import Enum


class RetrievalMode(str, Enum):
    """Retrieval pipeline used to rank candidate chunks."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid_reranked"
