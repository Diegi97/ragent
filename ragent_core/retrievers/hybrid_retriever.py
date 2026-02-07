import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import Dataset
from mxbai_rerank import MxbaiRerankV1, MxbaiRerankV2
from pylate import indexes, retrieve
from pylate import models as pylate_models

from ragent_core.retrievers import BM25Retriever, RetrievalResult
from ragent_core.retrievers.base import BaseRetriever
from ragent_core.utils import retry_on_oom

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 0
TOP_RERANK_CHUNKS = 50
CANDIDATES_PER_SYSTEM = 100
RRF_K = 60
INDEX_VERSION = "v2"


class HybridRetriever(BaseRetriever):
    """Two-stage retriever: BM25 + ColBERT candidates, then mxbai reranking."""

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
        self.bm25_stopwords = bm25_stopwords
        self.colbert_model_name = colbert_model_name
        self.reranker_model_name = reranker_model_name
        self.rerank_threshold = rerank_threshold
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.top_rerank_chunks = top_rerank_chunks
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._index_name = f"{dataset_name or 'default'}__{INDEX_VERSION}"

        self._bm25_retriever = BM25Retriever(
            dataset=dataset,
            dataset_name=dataset_name,
            bm25_stopwords=self.bm25_stopwords,
            override_index=override_index,
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
        )
        self._chunk_dataset = self._bm25_retriever.chunk_dataset

        self._init_reranker()
        self._init_colbert_index(override_index)

    @property
    def documents(self) -> dict:
        return self._bm25_retriever.documents

    def _init_reranker(self) -> None:
        logger.info("Loading mxbai reranker...")
        if "v1" in self.reranker_model_name:
            self._reranker = MxbaiRerankV1(self.reranker_model_name)
        else:
            self._reranker = MxbaiRerankV2(self.reranker_model_name)
        logger.info("mxbai reranker loaded.")

    def _init_colbert_index(self, override_index: bool) -> None:
        logger.info("Loading ColBERT model...")
        self._colbert_model = pylate_models.ColBERT(
            model_name_or_path=self.colbert_model_name,
            device=self.device,
        )
        logger.info("ColBERT model loaded.")

        self._colbert_index = indexes.PLAID(
            index_folder="pylate-index",
            index_name=self._index_name,
            override=override_index,
        )
        self._colbert_retriever = retrieve.ColBERT(index=self._colbert_index)

        should_index = override_index or not self._colbert_index._index.is_indexed
        if not should_index:
            logger.info("ColBERT chunk index already built.")
            return

        logger.info("Encoding and indexing chunks for ColBERT...")
        chunk_embeddings = self._colbert_model.encode(
            self._chunk_dataset["text"],
            is_query=False,
            batch_size=128,
            show_progress_bar=True,
        )
        self._colbert_index.add_documents(
            documents_ids=self._chunk_dataset["chunk_id"],
            documents_embeddings=chunk_embeddings,
        )
        logger.info("ColBERT chunk index built.")

    def _colbert_candidates(self, query: str, top_k: int) -> List[int]:
        query_embeddings = self._colbert_model.encode(
            [query],
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        )
        scores = self._colbert_retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=top_k,
        )
        if not scores:
            return []

        candidates: List[int] = []
        for result in scores[0]:
            chunk_id = int(result.get("id"))
            candidates.append(chunk_id)
        return candidates

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: Sequence[Sequence[int]],
        k: int = RRF_K,
    ) -> List[tuple[int, float]]:
        scores: Dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, chunk_id in enumerate(ranked_list):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    @retry_on_oom(min_batch_size=1)
    def _rerank_chunks(
        self,
        query: str,
        chunk_ids: Sequence[int],
        batch_size: int = 16,
        rerank_threshold: float = 1.0,
        show_progress: bool = False,
    ) -> List[RetrievalResult]:
        chunk_rows: List[dict[str, Any]] = []
        for chunk_index in chunk_ids:
            if 0 <= chunk_index < len(self._chunk_dataset):
                chunk_rows.append(self._chunk_dataset[chunk_index])

        if not chunk_rows:
            return []

        chunk_texts = [row.get("text", "") for row in chunk_rows]
        chunk_doc_ids = [row.get("doc_id") for row in chunk_rows]
        chunk_ids = [row.get("chunk_id") for row in chunk_rows]

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
                    chunk_id=chunk_ids[idx],
                    score=float(result.score),
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
        bm25_results = self._bm25_retriever.retrieve(
            query, top_k=CANDIDATES_PER_SYSTEM
        )
        bm25_ids = [result.chunk_id for result in bm25_results]
        logger.debug("BM25 chunk candidates: %d", len(bm25_results))

        colbert_ids = self._colbert_candidates(query, top_k=CANDIDATES_PER_SYSTEM)
        logger.debug("ColBERT chunk candidates: %d", len(colbert_ids))

        fused_candidates = self._reciprocal_rank_fusion([bm25_ids, colbert_ids])
        top_chunk_ids = [
            chunk_id for chunk_id, _ in fused_candidates[: self.top_rerank_chunks]
        ]
        logger.debug(
            "RRF merged chunk candidates (top %d): %d",
            self.top_rerank_chunks,
            len(top_chunk_ids),
        )

        reranked = self._rerank_chunks(
            query=query,
            chunk_ids=top_chunk_ids,
            batch_size=rerank_batch_size,
            rerank_threshold=self.rerank_threshold,
            show_progress=show_progress,
        )
        logger.debug("Reranked documents: %d", len(reranked))

        if top_k is None:
            return reranked
        return reranked[:top_k]
