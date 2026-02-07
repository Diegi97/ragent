import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

import bm25s
from datasets import Dataset

from ragent_core.retrievers import RetrievalResult
from ragent_core.retrievers.base import BaseRetriever
from ragent_core.retrievers.chunking import prepare_chunk_docs

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 0
DEFAULT_INDEX_ROOT = Path("bm25-index")
INDEX_VERSION = "v2"


class BM25Retriever(BaseRetriever):
    """BM25 retriever over fixed-size document chunks."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_name: Optional[str] = None,
        bm25_stopwords: Optional[str] = "en",
        chunk_dataset: Optional[Dataset] = None,
        override_index: bool = False,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        **kwargs,
    ) -> None:
        self.bm25_stopwords = bm25_stopwords
        self.override_index = override_index
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._documents = self._build_documents(dataset)

        self._index_dir = self._build_index_dir(dataset_name)

        if self._should_load_existing_index():
            chunk_docs, retriever = self._load_saved_index()
        else:
            chunk_docs = prepare_chunk_docs(
                chunk_dataset=chunk_dataset,
                dataset=dataset,
                chunk_size_tokens=self.chunk_size_tokens,
                chunk_overlap_tokens=self.chunk_overlap_tokens,
            )
            retriever = self._build_bm25_index(chunk_docs)
            self._save_retriever(retriever, self._index_dir)

        self._corpus = chunk_docs
        self._retriever = retriever
        self._chunk_dataset = Dataset.from_list(chunk_docs)

    def _build_index_dir(self, dataset_name: Optional[str]) -> str:
        safe_name = str(dataset_name or "default").replace("/", "__")
        return str(DEFAULT_INDEX_ROOT / f"{safe_name}__{INDEX_VERSION}")

    def _should_load_existing_index(self) -> bool:
        return not self.override_index and Path(self._index_dir).exists()

    def _load_saved_index(self) -> tuple[List[dict[str, Any]], bm25s.BM25]:
        logger.info("Loading BM25 index from '%s'...", self._index_dir)
        retriever = bm25s.BM25.load(self._index_dir, load_corpus=True)
        chunk_docs = retriever.corpus
        for idx, doc in enumerate(chunk_docs):
            doc["id"] = idx
        retriever.corpus = chunk_docs
        return chunk_docs, retriever

    def _build_bm25_index(self, chunk_docs: Sequence[dict[str, Any]]) -> bm25s.BM25:
        retriever = bm25s.BM25(corpus=chunk_docs)
        corpus_tokens = bm25s.tokenize(
            [doc.get("text", "") for doc in chunk_docs],
            stopwords=self.bm25_stopwords,
        )
        retriever.index(corpus_tokens)
        return retriever

    def _save_retriever(self, retriever: bm25s.BM25, index_dir: str) -> None:
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        retriever.save(index_dir)
        logger.info("Index successfully saved to '%s'.", index_dir)

    def _build_documents(
        self, dataset: Dataset
    ) -> dict:
        documents = {}
        for doc in dataset:
            documents[doc.get("id")] = doc 
        return documents

    @property
    def chunk_dataset(self) -> Dataset:
        return self._chunk_dataset

    @property
    def documents(self) -> dict:
        return self._documents

    def save(self, index_dir: str) -> None:
        if self._retriever is None:
            raise RuntimeError("Retriever not initialized.")
        self._save_retriever(self._retriever, index_dir)

    @classmethod
    def load(cls, index_dir: str, **kwargs) -> "BM25Retriever":
        logger.info("Loading retriever from '%s'...", index_dir)
        retriever_obj = bm25s.BM25.load(index_dir, load_corpus=True)
        dataset = Dataset.from_list(retriever_obj.corpus)
        return cls(dataset=dataset, chunk_dataset=dataset, **kwargs)

    def _retrieve_documents(
        self,
        query: str,
        top_k: int,
    ) -> List[tuple[dict[str, Any], float]]:
        if top_k <= 0:
            return []

        tokens = bm25s.tokenize(
            query,
            stopwords=self.bm25_stopwords,
            show_progress=False,
        )
        results, scores = self._retriever.retrieve(tokens, k=top_k, show_progress=False)
        return list(zip(results[0], scores[0]))

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        **kwargs,
    ) -> List[RetrievalResult]:
        results_with_scores = self._retrieve_documents(query, top_k=top_k)
        return [
            RetrievalResult(
                doc_id=doc.get("doc_id"),
                chunk_id=doc.get("chunk_id"),
                score=float(score),
                text=doc.get("text", ""),
            )
            for doc, score in results_with_scores
        ]
