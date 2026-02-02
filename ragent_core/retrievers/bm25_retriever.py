import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import bm25s
from datasets import Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

from ragent_core.retrievers import RetrievalResult
from ragent_core.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 0


class BM25Retriever(BaseRetriever):
    """
    Unified BM25 Retriever that builds or loads an index from a Dataset object.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_name: Optional[str] = None,
        bm25_stopwords: Optional[str] = "en",
        chunk_dataset: Optional[Dataset] = None,
        chunk_index_by_id: Optional[Dict[str, int]] = None,
        override_index: bool = False,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        **kwargs,
    ) -> None:
        """Initialize the BM25 retriever and build or load the index."""
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.bm25_stopwords = bm25_stopwords
        self._chunk_dataset = chunk_dataset
        self._chunk_index_by_id = chunk_index_by_id
        self.override_index = override_index
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._index_dir = self._build_index_dir(dataset_name)

        if not self.override_index and self._index_exists():
            self._load_from_index()
        else:
            self._init_chunker()
            if self._chunk_dataset is None:
                self._init_from_dataset(self.dataset)
            else:
                self._index_dataset(self._chunk_dataset)
            self.save(self._index_dir)

    def _build_index_dir(self, dataset_name: Optional[str]) -> str:
        """Return the dataset-specific on-disk index path."""
        dataset_name = dataset_name or "default"
        safe_name = str(dataset_name).replace("/", "__")
        index_name = f"{safe_name}"
        return str(Path("bm25-index") / index_name)

    def _index_exists(self) -> bool:
        """Check whether the BM25 index path exists on disk."""
        return Path(self._index_dir).exists()

    def _load_from_index(self) -> None:
        """Load a persisted BM25 index and corpus from disk."""
        logger.info("Loading BM25 index from '%s'...", self._index_dir)
        retriever_obj = bm25s.BM25.load(self._index_dir, load_corpus=True)
        corpus = retriever_obj.corpus
        self._corpus = self._normalize_corpus(corpus)
        self._retriever = retriever_obj
        self._chunk_dataset = Dataset.from_list(self._corpus)
        self._ensure_chunk_index_by_id(self._corpus)

    def _init_chunker(self) -> None:
        """Initialize the text chunker."""
        self._chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="o200k_base",
            chunk_size=self.chunk_size_tokens,
            chunk_overlap=self.chunk_overlap_tokens,
        )

    def _init_from_dataset(self, dataset: Dataset) -> None:
        """Chunk and index a dataset to initialize the BM25 retriever."""
        logger.info("Building in-memory BM25 chunk index...")
        self._chunk_dataset = self._build_chunk_dataset(dataset)
        self._index_dataset(self._chunk_dataset)
        logger.info("BM25 chunk index built.")

    def _index_dataset(self, chunk_dataset: Dataset) -> None:
        """Index a chunked dataset and build BM25 data structures."""
        corpus = self._normalize_corpus(chunk_dataset)
        self._corpus = corpus
        self._retriever = bm25s.BM25(corpus=corpus)
        corpus_tokens = bm25s.tokenize(
            [doc.get("text", "") for doc in corpus], stopwords=self.bm25_stopwords
        )
        self._retriever.index(corpus_tokens)
        self._ensure_chunk_index_by_id(corpus)

    def _ensure_chunk_index_by_id(self, corpus: Sequence[dict[str, Any]]) -> None:
        """Cache chunk ID to position mapping for fast lookup."""
        if self._chunk_index_by_id:
            return
        self._chunk_index_by_id = {}
        for idx, doc in enumerate(corpus):
            doc_id = doc.get("id")
            if doc_id is not None:
                self._chunk_index_by_id[str(doc_id)] = idx

    def _normalize_corpus(
        self, corpus: Dataset | Sequence[dict[str, Any]]
    ) -> List[dict[str, Any]]:
        """Normalize a dataset-like corpus into a list of dicts."""
        if hasattr(corpus, "to_list"):
            return corpus.to_list()
        if isinstance(corpus, list):
            return corpus
        return [item for item in corpus]

    def _build_chunk_dataset(self, dataset: Dataset) -> Dataset:
        """Chunk all documents and return a chunked Dataset."""
        chunk_docs: List[dict[str, Any]] = []

        for doc_index, doc in enumerate(
            tqdm(dataset, desc="Chunking documents", unit="doc")
        ):
            doc_id = doc.get("id", doc_index)
            title = doc.get("title", "")
            text = doc.get("text", "")
            chunks = self._chunker.split_text(text) if text else []
            if not chunks:
                chunks = [text]

            for chunk_index, chunk in enumerate(chunks):
                chunk_docs.append(
                    {
                        "id": f"{doc_id}::chunk_{chunk_index}",
                        "text": chunk,
                        "title": title,
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "doc_index": doc_index,
                    }
                )

        return Dataset.from_list(chunk_docs)

    def save(self, index_dir: str) -> None:
        """Persist the BM25 index and attached corpus to disk."""
        if self._retriever is None:
            raise RuntimeError("Retriever not initialized.")
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        self._retriever.save(index_dir)
        logger.info("Index successfully saved to '%s'.", index_dir)

    @classmethod
    def load(cls, index_dir: str, **kwargs) -> "BM25Retriever":
        """Load a BM25 retriever from a saved index directory."""
        logger.info("Loading retriever from '%s'...", index_dir)
        retriever_obj = bm25s.BM25.load(index_dir, load_corpus=True)
        dataset = Dataset.from_list(retriever_obj.corpus)
        return cls(dataset=dataset, chunk_dataset=dataset, **kwargs)

    def _iter_corpus(self):
        """Yield (index, document) pairs from the source dataset."""
        for idx, doc in enumerate(self.dataset):
            yield idx, doc

    def _get_doc_by_index(self, index: int):
        """Return a document by integer index."""
        return self.dataset[index]

    def _retrieve_documents(
        self,
        query: str,
        top_k: int,
    ) -> List[tuple[dict[str, Any], float]]:
        """Return (document, score) tuples for a query."""
        tokens = bm25s.tokenize(
            query, stopwords=self.bm25_stopwords, show_progress=False
        )
        results, scores = self._retriever.retrieve(tokens, k=top_k, show_progress=False)
        return list(zip(results[0], scores[0]))

    def retrieve_documents(
        self,
        query: str,
        top_k: int,
    ) -> List[tuple[dict[str, Any], float]]:
        """Public wrapper for retrieving (document, score) tuples."""
        return self._retrieve_documents(query, top_k=top_k)

    def candidate_ids(
        self,
        query: str,
        top_k: int,
    ) -> List[str]:
        """Return candidate chunk IDs for a query."""
        candidates: List[str] = []
        for doc, _score in self._retrieve_documents(query, top_k=top_k):
            doc_id = doc.get("id")
            if doc_id is not None:
                candidates.append(str(doc_id))
        return candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        bm25_k_candidates: Optional[int] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """Retrieve chunks for a query and return RetrievalResult objects."""
        if bm25_k_candidates is None:
            top_k = len(self._retriever.corpus) if top_k is None else top_k
        else:
            top_k = bm25_k_candidates

        results_with_scores = self._retrieve_documents(query, top_k=top_k)
        return [
            RetrievalResult(
                doc_id=doc.get("doc_id", doc.get("id")),
                score=float(score),
                text=doc.get("text", ""),
                document=doc,
            )
            for doc, score in results_with_scores
        ]
