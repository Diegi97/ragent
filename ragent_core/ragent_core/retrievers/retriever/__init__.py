import logging
import os
from typing import Any, Optional, Self

import numpy as np

from ragent_core.retrievers.base import BaseRetriever
from ragent_core.retrievers.catalog import (
    CATALOG_SCHEMA_VERSION,
    CorpusCatalogEntry,
    catalog_namespace,
)
from ragent_core.retrievers.document import Document, RetrievalResult
from ragent_core.retrievers.embedding import EmbeddingBackend
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.reranking import CrossEncoderReranker
from ragent_core.retrievers.service_clients.reranker import CrossEncoderServiceClient
from ragent_core.retrievers.settings import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LOGICAL_NAMESPACE,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_THRESHOLD,
    DEFAULT_RERANKER_MODEL_NAME,
    DEFAULT_TOP_K,
    DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
    NAMESPACE_PREFIX_ENV,
    SCAN_PAGE_SIZE,
    TOP_RERANK,
)
from ragent_core.retrievers.storage import (
    REMOTE_FIELDS,
    SOURCE_INDEX_FIELD,
    deserialize_document,
    deserialize_result,
)
from ragent_core.retrievers.transport import (
    create_turbopuffer_client,
    multi_query_rows,
    query_rows,
    turbopuffer_errors,
)

logger = logging.getLogger(__name__)


class TurbopufferRetriever(BaseRetriever):
    def __init__(
        self,
        reranker_model_name: Optional[str] = DEFAULT_RERANKER_MODEL_NAME,
        rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
        top_rerank: int = TOP_RERANK,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        *,
        client: Any = None,
        namespace: str = DEFAULT_LOGICAL_NAMESPACE,
        namespace_prefix: str | None = None,
    ) -> None:
        self.embedding: EmbeddingBackend | None = None
        self.client = client
        self.logical_namespace = namespace
        self.namespace_prefix = (
            namespace_prefix
            or os.getenv(NAMESPACE_PREFIX_ENV)
            or DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX
        )
        self.reranker: CrossEncoderReranker | None = None
        self.reranker_model_name = reranker_model_name
        self.rerank_threshold = rerank_threshold
        self.top_rerank = top_rerank
        self.rerank_batch_size = rerank_batch_size

    @classmethod
    def load_index(
        cls,
        namespace: str = DEFAULT_LOGICAL_NAMESPACE,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        reranker_model_name: Optional[str] = DEFAULT_RERANKER_MODEL_NAME,
        rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
        top_rerank: int = TOP_RERANK,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        max_seq_length: Optional[int] = None,
        embedding_service_url: Optional[str] = None,
        reranker_service_url: Optional[str] = None,
        load_embedding_backend: bool = True,
        *,
        client: Any = None,
        namespace_prefix: str | None = None,
    ) -> Self:
        retriever = cls(
            reranker_model_name=reranker_model_name,
            rerank_threshold=rerank_threshold,
            top_rerank=top_rerank,
            rerank_batch_size=rerank_batch_size,
            client=client or create_turbopuffer_client(),
            namespace=namespace,
            namespace_prefix=namespace_prefix,
        )
        if load_embedding_backend:
            retriever.embedding = EmbeddingBackend.load(
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
        return retriever

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = DEFAULT_TOP_K,
        retrieval_mode: Optional[RetrievalMode] = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        with turbopuffer_errors():
            del kwargs
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
                rows = self._hybrid_rows(
                    query, table_name, fetch_limit=min(top_k, SCAN_PAGE_SIZE)
                )
                return [deserialize_result(row) for row in rows[:top_k]]
            if self.reranker is None:
                raise RuntimeError(
                    "Hybrid reranked retrieval requires a loaded reranker."
                )
            fetch_limit = max(top_k, self.reranker.top_rerank or top_k)
            rows = self._hybrid_rows(query, table_name, fetch_limit=fetch_limit)
            reranked = self.reranker.rerank(query, rows)
            return [deserialize_result(row) for row in reranked[:top_k]]

    def retrieve_dense(
        self, query: str, table_name: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievalResult]:
        with turbopuffer_errors():
            entry = self._require_vectors(table_name)
            vector = self._encode_query(query)
            if len(vector) != entry.vector_dimensions:
                raise ValueError(
                    f"Query vector dimension {len(vector)} does not match cataloged "
                    f"dimension {entry.vector_dimensions}."
                )
            response = self.client.namespace(entry.chunks_namespace).query(
                rank_by=("vector", "ANN", vector.tolist()),
                top_k=top_k,
                include_attributes=list(REMOTE_FIELDS),
            )
            return [deserialize_result(row) for row in query_rows(response)]

    def retrieve_bm25(
        self, query: str, table_name: str, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievalResult]:
        with turbopuffer_errors():
            entry = self._catalog_entry(table_name)
            response = self.client.namespace(entry.chunks_namespace).query(
                rank_by=("content", "BM25", query),
                top_k=top_k,
                include_attributes=list(REMOTE_FIELDS),
            )
            return [deserialize_result(row) for row in query_rows(response)]

    def get_document(self, doc_id: int | str, table_name: str) -> Document | None:
        with turbopuffer_errors():
            response = self.get_documents_table(table_name).query(
                rank_by=("id", "asc"),
                filters=("id", "Eq", doc_id),
                top_k=1,
                include_attributes=list(REMOTE_FIELDS),
            )
            rows = query_rows(response)
            return deserialize_document(rows[0]) if rows else None

    def scan_chunks(
        self, table_name: str, server_regex: str
    ) -> list[tuple[str, int | str | None, str]]:
        with turbopuffer_errors():
            namespace = self.get_chunks_table(table_name)
            rows: list[tuple[str, int | str | None, str]] = []
            last_source_index: int | None = None
            while True:
                regex_filter: tuple[str, str, str] = (
                    "content",
                    "Regex",
                    server_regex,
                )
                filters: Any = regex_filter
                if last_source_index is not None:
                    filters = (
                        "And",
                        (regex_filter, (SOURCE_INDEX_FIELD, "Gt", last_source_index)),
                    )
                response = namespace.query(
                    rank_by=(SOURCE_INDEX_FIELD, "asc"),
                    filters=filters,
                    top_k=SCAN_PAGE_SIZE,
                    include_attributes=list(REMOTE_FIELDS),
                )
                page = query_rows(response)
                for row in page:
                    document = deserialize_document(row)
                    rows.append(
                        (document.content, document.document_id, document.title)
                    )
                if len(page) < SCAN_PAGE_SIZE:
                    break
                next_source_index = int(page[-1][SOURCE_INDEX_FIELD])
                if (
                    last_source_index is not None
                    and next_source_index <= last_source_index
                ):
                    raise RuntimeError("Turbopuffer scan pagination did not advance.")
                last_source_index = next_source_index
            return rows

    def find_catalog_entry(self, table_name: str) -> CorpusCatalogEntry | None:
        with turbopuffer_errors():
            catalog = self._catalog_namespace()
            if not catalog.exists():
                return None
            response = catalog.query(
                rank_by=("id", "asc"),
                filters=("id", "Eq", table_name),
                top_k=1,
                include_attributes=True,
            )
            rows = query_rows(response)
            return CorpusCatalogEntry.from_row(rows[0]) if rows else None

    def get_chunks_table(self, table_name: str) -> Any:
        with turbopuffer_errors():
            entry = self._catalog_entry(table_name)
            return self.client.namespace(entry.chunks_namespace)

    def get_documents_table(self, table_name: str) -> Any:
        with turbopuffer_errors():
            entry = self._catalog_entry(table_name)
            return self.client.namespace(entry.documents_namespace)

    def count_chunks(self, table_name: str) -> int:
        with turbopuffer_errors():
            return self._catalog_entry(table_name).chunk_count

    def get_chunk_by_source_index(
        self, table_name: str, source_index: int
    ) -> Document | None:
        with turbopuffer_errors():
            if source_index < 0:
                raise ValueError("source_index must be non-negative.")
            response = self.get_chunks_table(table_name).query(
                rank_by=(SOURCE_INDEX_FIELD, "asc"),
                filters=(SOURCE_INDEX_FIELD, "Eq", source_index),
                top_k=1,
                include_attributes=list(REMOTE_FIELDS),
            )
            rows = query_rows(response)
            return deserialize_document(rows[0]) if rows else None

    def _catalog_namespace(self) -> Any:
        name = catalog_namespace(
            self.logical_namespace, namespace_prefix=self.namespace_prefix
        )
        return self.client.namespace(name)

    def _catalog_entry(self, table_name: str) -> CorpusCatalogEntry:
        entry = self.find_catalog_entry(table_name)
        if entry is None:
            raise FileNotFoundError(
                f"No Turbopuffer catalog entry exists for "
                f"{self.logical_namespace}/{table_name}."
            )
        entry.require_identity(self.logical_namespace, table_name)
        if entry.schema_version != CATALOG_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported Turbopuffer catalog schema {entry.schema_version} "
                f"for {self.logical_namespace}/{table_name}; expected "
                f"{CATALOG_SCHEMA_VERSION}."
            )
        if not entry.ready:
            raise RuntimeError(
                f"Turbopuffer corpus {self.logical_namespace}/{table_name} is "
                "incomplete and cannot be used at runtime."
            )
        if (
            not self.client.namespace(entry.chunks_namespace).exists()
            or not self.client.namespace(entry.documents_namespace).exists()
        ):
            raise RuntimeError(
                f"Catalog entry for {self.logical_namespace}/{table_name} points "
                "to missing physical namespaces."
            )
        return entry

    def _load_reranker(
        self,
        reranker_model_name: str,
        device: Optional[str] = None,
        top_rerank: int = TOP_RERANK,
        rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        reranker_service_url: Optional[str] = None,
    ) -> None:
        if self.reranker is not None:
            return
        self.reranker_model_name = reranker_model_name
        ranker = (
            CrossEncoderServiceClient(reranker_service_url)
            if reranker_service_url is not None
            else None
        )
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model_name,
            device=device,
            top_rerank=top_rerank,
            rerank_threshold=rerank_threshold,
            batch_size=rerank_batch_size,
            ranker=ranker,
        )

    def _encode_query(self, query: str) -> np.ndarray:
        if self.embedding is None:
            raise RuntimeError("Embedding backend is not loaded")
        return self.embedding.encode_queries([query])[0]

    def _require_vectors(self, table_name: str) -> CorpusCatalogEntry:
        entry = self._catalog_entry(table_name)
        if not entry.vector_available:
            raise ValueError(
                f"Turbopuffer corpus {self.logical_namespace}/{table_name} is "
                "lexical-only; dense and hybrid retrieval require vectors."
            )
        if (
            self.embedding is not None
            and entry.embedding_model != self.embedding.model_name
        ):
            raise ValueError(
                f"Embedding model {self.embedding.model_name!r} does not match catalog model {entry.embedding_model!r}"
            )
        return entry

    def _hybrid_rows(
        self, query: str, table_name: str, *, fetch_limit: int
    ) -> list[dict[str, Any]]:
        entry = self._require_vectors(table_name)
        vector = self._encode_query(query)
        if len(vector) != entry.vector_dimensions:
            raise ValueError(
                f"Query vector dimension {len(vector)} does not match cataloged "
                f"dimension {entry.vector_dimensions}."
            )
        query_options = {
            "limit": min(fetch_limit, SCAN_PAGE_SIZE),
            "include_attributes": list(REMOTE_FIELDS),
        }
        response = self.client.namespace(entry.chunks_namespace).multi_query(
            queries=[
                {
                    **query_options,
                    "rank_by": ("vector", "ANN", vector.tolist()),
                },
                {
                    **query_options,
                    "rank_by": ("content", "BM25", query),
                },
            ],
            rerank_by=("RRF",),
        )
        return multi_query_rows(response)
