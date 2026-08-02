import gc
import importlib
import logging
import os
from datetime import timedelta
from typing import Any, Optional, Sequence

import lancedb
import numpy as np
import pyarrow as pa
import torch
from lancedb.rerankers import Reranker, RRFReranker
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

from ragent_core.retrievers.base import BaseRetriever
from ragent_core.retrievers.chunking import DOCUMENT_ID_KEY
from ragent_core.retrievers.document import (
    Document,
    DocumentLike,
    RetrievalResult,
    normalize_documents,
)
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.service_clients import (
    CrossEncoderServiceClient,
    EmbeddingServiceClient,
)

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
DEFAULT_RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"
DEFAULT_LOCAL_DB_URI = "lancedb/"
TOP_RERANK = 50
MIN_CHUNKS_FOR_ANN_INDEX = 100_000


def _flash_attention_available() -> bool:
    try:
        importlib.import_module("flash_attn")
    except Exception:
        return False
    return True


def _chunks_to_arrow_table(
    chunks: Sequence[Document], embeddings: np.ndarray
) -> pa.Table:
    """Build a LanceDB-ready Arrow table with an explicit float32 vector column.

    Vectors are written as a ``fixed_size_list<item: float32>[dim]`` column
    built directly from ``embeddings`` via ``FixedSizeListArray.from_arrays``,
    so LanceDB ingests them without ever materializing per-element Python
    floats (which a dict-of-lists path would do, spiking peak RAM to O(corpus)).
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks/emmb length mismatch: {len(chunks)} vs {embeddings.shape[0]}"
        )
    dim = embeddings.shape[1]
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(embeddings.reshape(-1), type=pa.float32()), dim
    )
    return pa.table(
        {
            "id": pa.array([chunk.id for chunk in chunks]),
            "title": pa.array([chunk.title for chunk in chunks], type=pa.string()),
            "content": pa.array([chunk.content for chunk in chunks], type=pa.string()),
            "metadata": pa.array([chunk.metadata for chunk in chunks]),
            "document_id": pa.array([chunk.document_id for chunk in chunks]),
            "vector": vector_array,
        }
    )


def _chunks_to_lexical_arrow_table(chunks: Sequence[Document]) -> pa.Table:
    """Build a LanceDB-ready chunk table without a vector column."""
    return pa.table(
        {
            "id": pa.array([chunk.id for chunk in chunks]),
            "title": pa.array([chunk.title for chunk in chunks], type=pa.string()),
            "content": pa.array([chunk.content for chunk in chunks], type=pa.string()),
            "metadata": pa.array([chunk.metadata for chunk in chunks]),
            "document_id": pa.array([chunk.document_id for chunk in chunks]),
        }
    )


def _connect_lancedb(namespace: str = "default") -> Any:
    """Connect to a local LanceDB instance.

    The ``namespace`` is appended as a sub-path to the base URI, so each
    namespace is an isolated LanceDB database (e.g. ``lancedb/default``).
    GCS is not involved here -- the retriever always reads from local disk.
    Use the standalone ``scripts/sync_lancedb_gcs.py`` to mirror a namespace
    to/from GCS (GCS is the durable source of truth, local disk is the hot
    read path).
    """
    base_uri = os.getenv("LANCEDB_DB_URI", DEFAULT_LOCAL_DB_URI)
    uri = f"{base_uri.rstrip('/')}/{namespace}"
    return lancedb.connect(uri=uri)


class CrossEncoderLanceDBReranker(Reranker):
    """LanceDB reranker using RRF fusion followed by CrossEncoder scoring."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        device: Optional[str] = None,
        top_rerank: int = TOP_RERANK,
        rerank_threshold: float = 0.0,
        batch_size: int = 8,
        max_length: int = 512,
        content_column: str = "content",
        return_score: str = "relevance",
        ranker: Any = None,
    ):
        super().__init__(return_score)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_rerank = top_rerank
        self.rerank_threshold = rerank_threshold
        self.batch_size = batch_size
        self.max_length = max_length
        self.content_column = content_column
        self._rrf_reranker = RRFReranker()

        if ranker is None:
            logger.info(
                "Loading CrossEncoder reranker '%s' with max_length=%d...",
                self.model_name,
                self.max_length,
            )
            self._reranker = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            logger.info("CrossEncoder reranker loaded.")
        else:
            self._reranker = ranker
            logger.info("Using remote CrossEncoder service for reranking.")

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        # First let LanceDB's RRF reranker fuse vector and FTS candidates,
        # then rerank that shortlist with the CrossEncoder.
        combined_results = self._rrf_reranker.rerank_hybrid(
            query, vector_results, fts_results
        )
        if self.top_rerank is not None:
            combined_results = combined_results.slice(0, self.top_rerank)
        return self._rerank_table(query, combined_results)

    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        return self._rerank_table(query, vector_results)

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        return self._rerank_table(query, fts_results)

    def _rerank_table(self, query: str, results: pa.Table) -> pa.Table:
        if len(results) == 0:
            return self._handle_empty_results(results)
        if self.content_column not in results.column_names:
            raise ValueError(
                f"Cannot rerank LanceDB results without '{self.content_column}' column"
            )

        rows = results.to_pylist()
        texts = [str(row.get(self.content_column) or "") for row in rows]
        logger.debug(
            "Reranking %d LanceDB candidates with batch size %d",
            len(texts),
            self.batch_size,
        )
        # CrossEncoder.rank returns [{"corpus_id", "score"}, ...] sorted by
        # score descending; corpus_id indexes the input ``texts`` list. We use
        # an identity activation to score raw logits (matching the previous
        # mxbai_rerank behavior that ``rerank_threshold`` was calibrated for);
        # the default ``nn.Sigmoid`` would squash scores to [0, 1] and break the
        # 3.0 threshold.
        rank_results = self._reranker.rank(
            query,
            texts,
            top_k=len(texts),
            return_documents=False,
            batch_size=self.batch_size,
            show_progress_bar=True,
            activation_fn=torch.nn.Identity(),
        )

        selected_indices: list[int] = []
        relevance_scores: list[float] = []
        for result in rank_results:
            if result["score"] < self.rerank_threshold:
                continue
            selected_indices.append(int(result["corpus_id"]))
            relevance_scores.append(float(result["score"]))

        reranked = results.take(pa.array(selected_indices, type=pa.int64()))
        reranked = self._replace_column(
            reranked,
            "_relevance_score",
            pa.array(relevance_scores, type=pa.float32()),
        )
        if self.score == "relevance":
            reranked = self._keep_relevance_score(reranked)
        return reranked

    @staticmethod
    def _replace_column(table: pa.Table, name: str, column: pa.Array) -> pa.Table:
        if name in table.column_names:
            index = table.column_names.index(name)
            return table.set_column(index, name, column)
        return table.append_column(name, column)


class LanceDBRetriever(BaseRetriever):
    def __init__(
        self,
        reranker_model_name: Optional[str] = DEFAULT_RERANKER_MODEL_NAME,
        rerank_threshold: float = 0.0,
        top_rerank: int = TOP_RERANK,
        rerank_batch_size: int = 8,
    ):
        self._model: Optional[SentenceTransformer] = None
        self._embedding_client: Optional[EmbeddingServiceClient] = None
        self.db: Any = None
        self.chunks_table: Any = None
        self.documents_table: Any = None
        self.reranker: Any = None
        self.embedding_model_name = DEFAULT_EMBEDDING_MODEL_NAME
        self.reranker_model_name = reranker_model_name

    @classmethod
    def build_index(
        cls,
        chunks: Optional[Sequence[DocumentLike]] = None,
        table_name: str = "",
        namespace: str = "default",
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = True,
        documents: Optional[Sequence[DocumentLike]] = None,
        max_seq_length: Optional[int] = None,
        min_chunks_for_ann_index: int = MIN_CHUNKS_FOR_ANN_INDEX,
        embedding_service_url: Optional[str] = None,
        build_embeddings: bool = True,
    ):
        """Build a LanceDB retriever from chunks and documents on local disk.

        Args:
            chunks: Chunk rows to index. Each chunk is expected to carry a
                top-level ``document_id`` (set by :func:`chunk_documents`).
            documents: The originating (non-chunked) documents. When provided,
                a separate ``<table_name>_documents`` table is created with one
                row per document and a scalar index on ``id``, so
                :meth:`AgentRetriever.get_document` can fetch a full document
                by id without scanning the (much larger) chunks table or
                reading document content duplicated into chunk metadata.
            build_embeddings: Whether to encode chunks and include a vector
                column. When false, only the lexical FTS and scalar indexes are
                built.

        The index is always built locally. To persist it to GCS (the durable
        source of truth) use the standalone ``scripts/sync_lancedb_gcs.py``
        script after building; the retriever itself never touches GCS.
        """
        if chunks is None:
            raise ValueError("build_index requires chunks to index.")
        retriever = cls()
        retriever.db = _connect_lancedb(namespace)

        normalized_chunks = normalize_documents(chunks)
        if build_embeddings:
            retriever._load_embedding_backend(
                model_name=model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                max_seq_length=max_seq_length,
                embedding_service_url=embedding_service_url,
            )
            normalized_chunks, embeddings = retriever._encode_documents(
                normalized_chunks,
                batch_size,
                show_progress_bar,
            )
            chunks_table_data = _chunks_to_arrow_table(
                normalized_chunks,
                embeddings,
            )
        else:
            logger.info("Skipping chunk embeddings; building lexical indexes only.")
            chunks_table_data = _chunks_to_lexical_arrow_table(normalized_chunks)

        retriever.chunks_table = retriever.db.create_table(
            f"{table_name}_chunks", data=chunks_table_data, mode="overwrite"
        )
        retriever.chunks_table.create_fts_index(
            "content", replace=True, name="content_idx"
        )
        retriever.chunks_table.wait_for_index(["content_idx"])
        # Scalar index on the top-level document_id column so filtered chunk
        # lookups by source document are indexed rather than full scans.
        if normalized_chunks and any(
            chunk.document_id is not None for chunk in normalized_chunks
        ):
            retriever.chunks_table.create_scalar_index(
                "document_id", replace=True, name="document_id_idx"
            )
        if not build_embeddings:
            logger.info("Skipping ANN vector index for lexical-only table.")
        elif len(normalized_chunks) >= min_chunks_for_ann_index:
            logger.info(
                "Building ANN vector index for %d chunks (threshold=%d).",
                len(normalized_chunks),
                min_chunks_for_ann_index,
            )
            retriever.chunks_table.create_index(
                vector_column_name="vector",
                index_type="IVF_HNSW_FLAT",
                replace=True,
                name="vector_idx",
                accelerator="cuda" if device == "cuda" else None,
                metric="cosine",
            )
            retriever.chunks_table.wait_for_index(
                ["vector_idx"], timeout=timedelta(seconds=1200)
            )
        else:
            logger.info(
                "Skipping ANN vector index for %d chunks (threshold=%d); "
                "LanceDB will use exact vector search.",
                len(normalized_chunks),
                min_chunks_for_ann_index,
            )

        # Documents table: one row per source document, scalar-indexed by id.
        # This is what makes get_document/read_tool fast -- a point lookup
        # here replaces a full scan of the chunks table that previously had
        # to read the document content duplicated into every chunk's metadata.
        if documents is not None:
            normalized_documents = normalize_documents(documents)
            document_dicts = [doc.to_dict() for doc in normalized_documents]
            retriever.documents_table = retriever.db.create_table(
                f"{table_name}_documents", data=document_dicts, mode="overwrite"
            )
            retriever.documents_table.create_scalar_index(
                "id", replace=True, name="id_idx"
            )

    @classmethod
    def load_index(
        cls,
        namespace: str = "default",
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        reranker_model_name: Optional[str] = DEFAULT_RERANKER_MODEL_NAME,
        rerank_threshold: float = 0.0,
        top_rerank: int = TOP_RERANK,
        rerank_batch_size: int = 8,
        max_seq_length: Optional[int] = None,
        embedding_service_url: Optional[str] = None,
        reranker_service_url: Optional[str] = None,
        load_embedding_backend: bool = True,
    ) -> "LanceDBRetriever":
        """Load an index, optionally with a learned CrossEncoder reranker.

        Pass ``reranker_model_name=None`` to skip loading the CrossEncoder.
        Hybrid retrieval will then use reciprocal-rank fusion only.
        Pass ``load_embedding_backend=False`` for lexical-only consumers that
        need a LanceDB connection without a query embedding model or service.
        """
        retriever = cls(
            reranker_model_name=reranker_model_name,
            rerank_threshold=rerank_threshold,
            top_rerank=top_rerank,
            rerank_batch_size=rerank_batch_size,
        )
        if load_embedding_backend:
            retriever._load_embedding_backend(
                model_name=model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                max_seq_length=max_seq_length,
                embedding_service_url=embedding_service_url,
            )
        if reranker_service_url is not None or reranker_model_name is not None:
            retriever._load_reranker(
                reranker_model_name=reranker_model_name or DEFAULT_RERANKER_MODEL_NAME,
                device=device,
                top_rerank=top_rerank,
                rerank_threshold=rerank_threshold,
                rerank_batch_size=rerank_batch_size,
                reranker_service_url=reranker_service_url,
            )
        retriever.db = _connect_lancedb(namespace)
        return retriever

    def _load_embedding_backend(
        self,
        model_name: str,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        max_seq_length: Optional[int] = None,
        embedding_service_url: Optional[str] = None,
    ) -> None:
        self.embedding_model_name = model_name
        if embedding_service_url is not None:
            self._embedding_client = EmbeddingServiceClient(
                embedding_service_url,
            )
            logger.info("Using remote embedding service at %s.", embedding_service_url)
            return
        self._load_model(model_name, device, trust_remote_code, max_seq_length)

    def _load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        max_seq_length: Optional[int] = None,
    ):
        if self._model is None:
            logger.info("Loading embedding model '%s'...", model_name)
            model_device = device
            model_kwargs: dict[str, Any] = {"dtype": "auto"}
            if (
                device in (None, "cuda")
                and torch.cuda.is_available()
                and _flash_attention_available()
            ):
                model_device = "cuda"
                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                }
                logger.info("Using flash_attention_2 for embedding model.")
            self._model = SentenceTransformer(
                model_name,
                device=model_device,
                trust_remote_code=trust_remote_code,
                model_kwargs=model_kwargs,
            )
            if max_seq_length is not None:
                self._model.max_seq_length = max_seq_length
                logger.info("Set embedding model max_seq_length=%d", max_seq_length)
            self.embedding_model_name = model_name
            logger.info("Embedding model loaded.")

    def _load_reranker(
        self,
        reranker_model_name: str,
        device: Optional[str] = None,
        top_rerank: int = TOP_RERANK,
        rerank_threshold: float = 0.0,
        rerank_batch_size: int = 8,
        reranker_service_url: Optional[str] = None,
    ):
        if self.reranker is None:
            self.reranker_model_name = reranker_model_name
            ranker = (
                CrossEncoderServiceClient(
                    reranker_service_url,
                )
                if reranker_service_url is not None
                else None
            )
            self.reranker = CrossEncoderLanceDBReranker(
                model_name=reranker_model_name,
                device=device,
                top_rerank=top_rerank,
                rerank_threshold=rerank_threshold,
                batch_size=rerank_batch_size,
                ranker=ranker,
            )

    def _encode_documents(
        self,
        documents: Sequence[DocumentLike],
        batch_size: int,
        show_progress_bar: bool,
    ) -> tuple[list[Document], np.ndarray]:
        normalized_documents = normalize_documents(documents)
        if not normalized_documents:
            return [], np.empty((0, 0), dtype=np.float32)

        logger.info(
            "Encoding %d documents for exact embedding search...",
            len(normalized_documents),
        )

        if self._embedding_client is not None:
            batched_embeddings = []
            for start in tqdm(
                range(0, len(normalized_documents), batch_size),
                desc="Encoding documents",
                disable=not show_progress_bar,
                unit="batch",
            ):
                texts = [
                    document.content
                    for document in normalized_documents[start : start + batch_size]
                ]
                batched_embeddings.append(
                    self._embedding_client.encode_documents(texts)
                )
            return normalized_documents, np.concatenate(batched_embeddings, axis=0)

        batch_starts = range(0, len(normalized_documents), batch_size)
        cuda_available = torch.cuda.is_available()
        # Per-call (micro) batch size handed to model.encode; halved on OOM and
        # kept at the lower value for the rest of the run.
        current_encode_batch = batch_size
        min_encode_batch = 1
        batched_embeddings: list[np.ndarray] = []

        with torch.inference_mode():
            for start in tqdm(
                batch_starts,
                desc="Encoding documents",
                disable=not show_progress_bar,
                unit="batch",
            ):
                batch_documents = normalized_documents[start : start + batch_size]
                texts = [doc.content for doc in batch_documents]

                while True:
                    try:
                        batch_embeddings = self._model.encode(
                            texts,
                            batch_size=current_encode_batch,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        if cuda_available:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        gc.collect()
                        if current_encode_batch <= min_encode_batch:
                            logger.error(
                                "CUDA OOM at minimum batch_size=%d; cannot recover.",
                                current_encode_batch,
                            )
                            raise
                        new_batch = max(min_encode_batch, current_encode_batch // 2)
                        logger.warning(
                            "CUDA OOM during encode (batch_size=%d). "
                            "Retrying with batch_size=%d.",
                            current_encode_batch,
                            new_batch,
                        )
                        current_encode_batch = new_batch

                # Keep embeddings as a compact float32 numpy array end-to-end;
                # never box them into Python floats (the .tolist() path) so
                # ingestion peak memory stays O(batch) rather than O(corpus).
                batched_embeddings.append(
                    np.ascontiguousarray(batch_embeddings, dtype=np.float32)
                )
                del batch_embeddings, batch_documents, texts

                gc.collect()
                if cuda_available:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        if not batched_embeddings:
            return normalized_documents, np.empty((0, 0), dtype=np.float32)
        embeddings = np.concatenate(batched_embeddings, axis=0)
        del batched_embeddings
        gc.collect()
        if cuda_available:
            torch.cuda.empty_cache()
        return normalized_documents, embeddings

    def _encode_query(self, query: str) -> np.ndarray:
        if self._embedding_client is not None:
            return self._embedding_client.encode_queries([query])[0]
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded.")
        embedding = self._model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            prompt_name="web_search_query",
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(embedding, dtype=np.float32)[0]

    def get_chunks_table(self, table_name: str) -> Any:
        if self.db is None:
            raise RuntimeError("LanceDB database is not loaded.")
        return self.db.open_table(f"{table_name}_chunks")

    def get_documents_table(self, table_name: str) -> Any:
        if self.db is None:
            raise RuntimeError("LanceDB database is not loaded.")
        return self.db.open_table(f"{table_name}_documents")

    def retrieve_dense(
        self,
        query: str,
        table_name: str,
        top_k: int = 50,
    ) -> list[RetrievalResult]:
        """Retrieve chunk-level dense vector matches without hybrid fusion."""
        chunks_table = self.get_chunks_table(table_name)
        query_embedding = self._encode_query(query)
        results = (
            chunks_table.search(
                query_embedding,
                vector_column_name="vector",
                query_type="vector",
            )
            .limit(top_k)
            .select(["id", "title", "content", "metadata", "document_id", "_distance"])
            .to_list()
        )
        return [self._row_to_result(row) for row in results]

    def retrieve_bm25(
        self,
        query: str,
        table_name: str,
        top_k: int = 50,
    ) -> list[RetrievalResult]:
        """Retrieve chunk-level lexical matches from LanceDB full-text search."""
        chunks_table = self.get_chunks_table(table_name)
        results = (
            chunks_table.search(
                query,
                query_type="fts",
                fts_columns="content",
            )
            .limit(top_k)
            .select(["id", "title", "content", "metadata", "document_id", "_score"])
            .to_list()
        )
        return [self._row_to_result(row) for row in results]

    def _retrieve_hybrid(
        self,
        query: str,
        table_name: str,
        top_k: int,
        reranker: Reranker,
        fetch_limit: int,
    ) -> list[RetrievalResult]:
        chunks_table = self.get_chunks_table(table_name)
        query_embedding = self._encode_query(query)
        results = (
            chunks_table.search(
                query_type="hybrid",
                fts_columns="content",
            )
            .vector(query_embedding)
            .text(query)
            .rerank(reranker)
            .limit(fetch_limit)
            .select(["id", "title", "content", "metadata", "document_id"])
            .to_list()
        )
        return [self._row_to_result(row) for row in results[:top_k]]

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = 50,
        retrieval_mode: Optional[RetrievalMode] = None,
        **kwargs,
    ) -> list[RetrievalResult]:
        """Retrieve chunks using the selected retrieval pipeline.

        When no mode is supplied, preserve the historical behavior: hybrid
        retrieval uses the learned reranker when one is loaded and otherwise
        falls back to reciprocal-rank fusion.
        """
        mode = (
            RetrievalMode(retrieval_mode)
            if retrieval_mode is not None
            else (
                RetrievalMode.HYBRID_RERANKED
                if self.reranker is not None
                else RetrievalMode.HYBRID
            )
        )
        if mode is RetrievalMode.BM25:
            return self.retrieve_bm25(query, table_name=table_name, top_k=top_k)
        if mode is RetrievalMode.DENSE:
            return self.retrieve_dense(query, table_name=table_name, top_k=top_k)
        if mode is RetrievalMode.HYBRID:
            return self._retrieve_hybrid(
                query,
                table_name=table_name,
                top_k=top_k,
                reranker=RRFReranker(),
                fetch_limit=top_k,
            )
        if self.reranker is None:
            raise RuntimeError("Hybrid reranked retrieval requires a loaded reranker.")
        fetch_limit = max(top_k, self.reranker.top_rerank or top_k)
        return self._retrieve_hybrid(
            query,
            table_name=table_name,
            top_k=top_k,
            reranker=self.reranker,
            fetch_limit=fetch_limit,
        )

    @staticmethod
    def _row_to_document(row: dict[str, Any]) -> Document:
        return Document(
            id=row["id"],
            title=row.get("title") or "",
            content=row.get("content") or "",
            metadata=dict(row.get("metadata") or {}),
            document_id=row.get("document_id"),
        )

    @staticmethod
    def _row_to_result(row: dict[str, Any]) -> RetrievalResult:
        score = (
            row.get("_relevance_score")
            or row.get("_score")
            or row.get("_distance")
            or 0.0
        )
        metadata = dict(row.get("metadata") or {})
        document_id = row.get("document_id")
        if document_id is not None:
            metadata.setdefault(DOCUMENT_ID_KEY, document_id)
        return RetrievalResult(
            id=row["id"],
            title=row.get("title") or "",
            content=row.get("content") or "",
            metadata=metadata,
            score=float(score),
        )
