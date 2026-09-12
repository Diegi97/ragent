from enum import Enum


class RetrievalMode(str, Enum):
    """Retrieval pipeline used to rank candidate chunks."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid_reranked"

    @property
    def needs_embeddings(self) -> bool:
        return self is not self.BM25

    @property
    def needs_reranker(self) -> bool:
        return self is self.HYBRID_RERANKED
