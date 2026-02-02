import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import Dataset
from mxbai_rerank import MxbaiRerankV1, MxbaiRerankV2
from pylate import indexes, retrieve
from pylate import models as pylate_models

from ragent_core.utils import retry_on_oom
from ragent_core.retrievers.base import BaseRetriever
from ragent_core.retrievers import BM25Retriever, RetrievalResult

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 0
TOP_RERANK_CHUNKS = 50


class HybridRetriever(BaseRetriever):
    """
    High-quality two-stage retriever that indexes chunks instead of full documents:
      1) Candidate generation = BM25Retriever + ColBERT (pylate) over fixed-size chunks
      2) Reranking = Mixedbread mxbai-rerank-v2 on the top chunk candidates

    Notes:
    - Uses Reciprocal Rank Fusion (RRF) to merge BM25 and ColBERT results
    - Passes top 50 chunk candidates from RRF to the reranker
    - Returns reranked chunks directly (multiple chunks per document allowed)
    - Requires optional dependencies: pylate (for ColBERT) and mxbai-rerank (for reranking)
    - Does not depend on repository's BM25Client
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_name: Optional[str] = None,
        bm25_stopwords: Optional[str] = "english",
        colbert_model_name: str = "lightonai/GTE-ModernColBERT-v1",
        reranker_model_name: str = "mixedbread-ai/mxbai-rerank-base-v2",
        rerank_threshold: float = 3.0,
        device: Optional[str] = None,
        override_index: bool = False,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        top_rerank_chunks: int = TOP_RERANK_CHUNKS,
        **kwargs,
    ) -> None:
        """Construct the hybrid chunked retriever and build BM25, ColBERT, and reranker assets."""
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.bm25_stopwords = bm25_stopwords
        self.colbert_model_name = colbert_model_name
        self.reranker_model_name = reranker_model_name
        self.rerank_threshold = rerank_threshold
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.top_rerank_chunks = top_rerank_chunks
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._init_document_index()
        self._init_reranker()
        self._bm25_retriever = BM25Retriever(
            dataset=self.dataset,
            dataset_name=self.dataset_name,
            bm25_stopwords=self.bm25_stopwords,
            override_index=override_index,
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
        )
        self._chunk_dataset = self._bm25_retriever._chunk_dataset
        self._chunk_index_by_id = self._bm25_retriever._chunk_index_by_id

        logger.info(
            "Chunked %d documents into %d chunks.",
            len(self.dataset),
            len(self._chunk_dataset),
        )
        self._init_colbert_index(override_index)

    def _init_document_index(self) -> None:
        """Build document ID mappings."""
        if "id" in self.dataset.column_names:
            self.doc_ids = list(self.dataset["id"])
        else:
            self.doc_ids = list(range(len(self.dataset)))
        self._doc_index_by_id: Dict[Any, int] = {
            doc_id: idx for idx, doc_id in enumerate(self.doc_ids)
        }

    def _init_reranker(self) -> None:
        """Load the mxbai reranker model."""
        logger.info("Loading mxbai reranker...")
        if "v1" in self.reranker_model_name:
            self._reranker = MxbaiRerankV1(self.reranker_model_name)
        else:
            self._reranker = MxbaiRerankV2(self.reranker_model_name)
        logger.info("mxbai reranker loaded.")

    def _init_colbert_index(self, override_index: bool) -> None:
        """Load ColBERT model and build/load the ColBERT index."""
        logger.info("Loading ColBERT model...")
        self._colbert_model = pylate_models.ColBERT(
            model_name_or_path=self.colbert_model_name, device=self.device
        )
        logger.info("ColBERT model loaded.")

        index_name = self.dataset_name or "default"

        logger.info("Building ColBERT chunk index...")
        self._colbert_index = indexes.PLAID(
            index_folder="pylate-index",
            index_name=index_name,
            override=override_index,
        )
        self._colbert_retriever = retrieve.ColBERT(index=self._colbert_index)

        if not self._colbert_index._index.is_indexed or override_index:
            logger.info("Index not built, encoding and indexing chunks for ColBERT")
            chunk_embeddings = self._colbert_model.encode(
                self._chunk_dataset["text"],
                is_query=False,
                batch_size=128,
                show_progress_bar=True,
            )
            self._colbert_index.add_documents(
                documents_ids=self._chunk_dataset["id"],
                documents_embeddings=chunk_embeddings,
            )
            logger.info("ColBERT chunk index built.")
        else:
            logger.info("ColBERT chunk index already built.")

    def _iter_corpus(self):
        for idx in range(len(self.dataset)):
            yield idx, self.dataset[idx]

    def _get_doc_by_index(self, index: int):
        return self.dataset[index]

    def _bm25_candidates(self, query: str, bm25_k_candidates: int) -> List[str]:
        """Return BM25 candidate chunk identifiers for a query using the configured stopwords."""
        candidate_ids = self._bm25_retriever.candidate_ids(
            query, top_k=bm25_k_candidates
        )
        return [cid for cid in candidate_ids if cid in self._chunk_index_by_id]

    def _colbert_candidates(self, query: str, colbert_k_candidates: int) -> List[str]:
        """Return ColBERT candidate chunk identifiers for a query."""
        if self._colbert_retriever is None or self._colbert_model is None:
            return []

        query_embeddings = self._colbert_model.encode(
            [query],
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        )

        scores = self._colbert_retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=colbert_k_candidates,
        )

        if not scores:
            return []

        results = scores[0]
        candidates: List[str] = []
        for result in results:
            chunk_id = result.get("id")
            if isinstance(chunk_id, str) and chunk_id in self._chunk_index_by_id:
                candidates.append(chunk_id)
        return candidates

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[str]],
        k: int = 60,
    ) -> List[tuple[str, float]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            ranked_lists: List of ranked candidate lists (ordered by relevance)
            k: RRF constant (default 60 is typical)

        Returns:
            List of (chunk_id, rrf_score) tuples sorted by score descending
        """
        rrf_scores: Dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, chunk_id in enumerate(ranked_list):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (
                    k + rank + 1
                )

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    @retry_on_oom(min_batch_size=1)
    def _rerank_chunks(
        self,
        query: str,
        chunk_ids: Sequence[str],
        batch_size: int = 16,
        rerank_threshold: float = 1.0,
        show_progress: bool = False,
    ) -> List[RetrievalResult]:
        """Score candidate chunks with the mixedbread reranker and return filtered results."""
        if not chunk_ids:
            return []

        # Resolve chunk IDs to dataset indices
        valid_indices = [
            self._chunk_index_by_id[cid]
            for cid in chunk_ids
            if cid in self._chunk_index_by_id
        ]
        if not valid_indices:
            return []

        chunk_texts = [self._chunk_dataset[i]["text"] for i in valid_indices]
        chunk_doc_ids = [self._chunk_dataset[i]["doc_id"] for i in valid_indices]

        rank_results = self._reranker.rank(
            query,
            chunk_texts,
            return_documents=False,
            top_k=len(chunk_texts),
            batch_size=batch_size,
            show_progress=show_progress,
        )

        logger.debug("Applying rerank threshold: %.3f", rerank_threshold)

        reranked: List[RetrievalResult] = []
        for result in rank_results:
            if result.score < rerank_threshold:
                continue
            idx = result.index
            reranked.append(
                RetrievalResult(
                    doc_id=chunk_doc_ids[idx],
                    score=result.score,
                    text=chunk_texts[idx].strip(),
                )
            )

        return reranked

    def retrieve(
        self,
        query: str,
        rerank_batch_size: int = 16,
        top_k: Optional[int] = 50,
        show_progress: bool = False,
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks for `query` using RRF to merge BM25 and ColBERT chunk results.

        Args:
            query: Natural-language string to retrieve against.
            rerank_batch_size: Batch size for the mixedbread reranker; increase cautiously to avoid OOM.
            top_k: Hard cap on the number of reranked chunks returned; `None` disables truncation.
            show_progress: Whether to show progress bars during retrieval.
        """
        candidates_per_system = 100
        bm25_ids = self._bm25_candidates(query, bm25_k_candidates=candidates_per_system)
        logger.debug("BM25 chunk candidates: %d", len(bm25_ids))
        colbert_ids = self._colbert_candidates(
            query, colbert_k_candidates=candidates_per_system
        )
        logger.debug("ColBERT chunk candidates: %d", len(colbert_ids))

        rrf_candidates = self._reciprocal_rank_fusion([bm25_ids, colbert_ids], k=60)
        top_chunk_ids = [
            chunk_id for chunk_id, _ in rrf_candidates[: self.top_rerank_chunks]
        ]
        logger.debug(
            "RRF merged chunk candidates (top %d): %d",
            self.top_rerank_chunks,
            len(top_chunk_ids),
        )

        reranked = self._rerank_chunks(
            query,
            top_chunk_ids,
            batch_size=rerank_batch_size,
            rerank_threshold=self.rerank_threshold,
            show_progress=show_progress,
        )
        logger.debug("Reranked documents: %d", len(reranked))
        if top_k is None:
            return reranked
        return reranked[:top_k]
