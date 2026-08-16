import importlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Self

import numpy as np
import torch
import turbopuffer
from sentence_transformers import CrossEncoder, SentenceTransformer
from turbopuffer.types.namespace_multi_query_response import (
    NamespaceMultiQueryResponse,
)
from turbopuffer.types.namespace_query_response import NamespaceQueryResponse

from ragent_core.retrievers.base import BaseRetriever
from ragent_core.retrievers.chunking import DOCUMENT_ID_KEY
from ragent_core.retrievers.document import Document, RetrievalResult
from ragent_core.retrievers.mode import RetrievalMode
from ragent_core.retrievers.service_clients import (
    CrossEncoderServiceClient,
    EmbeddingServiceClient,
)

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
DEFAULT_RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-base-v2"
DEFAULT_TURBOPUFFER_REGION = "gcp-us-central1"
DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX = "ragent"
DEFAULT_LOGICAL_NAMESPACE = "default"
TOP_RERANK = 50

CATALOG_SCHEMA_VERSION = 1
SCAN_PAGE_SIZE = 10_000

_REMOTE_FIELDS = (
    "title",
    "content",
    "metadata_json",
    "document_id",
    "source_index",
)


def _flash_attention_available() -> bool:
    try:
        importlib.import_module("flash_attn")
    except Exception:
        return False
    return True


def catalog_namespace(
    logical_namespace: str = DEFAULT_LOGICAL_NAMESPACE,
    namespace_prefix: str = DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
) -> str:
    return f"{namespace_prefix}.{logical_namespace}.catalog"


def corpus_namespaces(
    table_name: str,
    logical_namespace: str = DEFAULT_LOGICAL_NAMESPACE,
    namespace_prefix: str = DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
) -> tuple[str, str]:
    return (
        f"{namespace_prefix}.{logical_namespace}.{table_name}.chunks",
        f"{namespace_prefix}.{logical_namespace}.{table_name}.documents",
    )


def create_turbopuffer_client(
    *,
    api_key: str | None = None,
    region: str | None = None,
) -> Any:
    resolved_region = (
        region or os.getenv("TURBOPUFFER_REGION") or DEFAULT_TURBOPUFFER_REGION
    )
    options: dict[str, Any] = {
        "region": resolved_region,
        "timeout": 60.0,
        "max_retries": 4,
    }
    if api_key is not None:
        options["api_key"] = api_key
    return turbopuffer.Turbopuffer(**options)


def _query_rows(response: NamespaceQueryResponse) -> list[dict[str, Any]]:
    return [row.model_dump(by_alias=True) for row in response.rows or []]


def _multi_query_rows(
    response: NamespaceMultiQueryResponse,
) -> list[dict[str, Any]]:
    return [
        row.model_dump(by_alias=True)
        for result in response.results
        for row in result.rows or []
    ]


@dataclass(frozen=True)
class CorpusCatalogEntry:
    table_name: str
    logical_namespace: str
    chunks_namespace: str
    documents_namespace: str
    schema_version: int
    chunk_count: int
    document_count: int
    vector_available: bool
    vector_dimensions: int
    embedding_model: str
    ready: bool

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        return cls(
            table_name=str(row["table_name"]),
            logical_namespace=str(row["logical_namespace"]),
            chunks_namespace=str(row["chunks_namespace"]),
            documents_namespace=str(row["documents_namespace"]),
            schema_version=int(row["schema_version"]),
            chunk_count=int(row["chunk_count"]),
            document_count=int(row["document_count"]),
            vector_available=bool(row["vector_available"]),
            vector_dimensions=int(row["vector_dimensions"]),
            embedding_model=str(row.get("embedding_model") or ""),
            ready=bool(row["ready"]),
        )


class CrossEncoderReranker:
    """Apply raw-logit CrossEncoder scoring to a retrieval shortlist."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        device: Optional[str] = None,
        top_rerank: int = TOP_RERANK,
        rerank_threshold: float = 0.0,
        batch_size: int = 8,
        max_length: int = 512,
        ranker: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_rerank = top_rerank
        self.rerank_threshold = rerank_threshold
        self.batch_size = batch_size
        self.max_length = max_length
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
        else:
            self._reranker = ranker
            logger.info("Using remote CrossEncoder service for reranking.")

    def rerank(
        self, query: str, rows: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        candidates = list(rows[: self.top_rerank])
        if not candidates:
            return []
        texts = [str(row.get("content") or "") for row in candidates]
        rank_results = self._reranker.rank(
            query,
            texts,
            top_k=len(texts),
            return_documents=False,
            batch_size=self.batch_size,
            show_progress_bar=True,
            activation_fn=torch.nn.Identity(),
        )
        selected: list[dict[str, Any]] = []
        for result in rank_results:
            score = float(result["score"])
            if score < self.rerank_threshold:
                continue
            row = dict(candidates[int(result["corpus_id"])])
            row["_relevance_score"] = score
            selected.append(row)
        return selected


class TurbopufferRetriever(BaseRetriever):
    def __init__(
        self,
        reranker_model_name: Optional[str] = DEFAULT_RERANKER_MODEL_NAME,
        rerank_threshold: float = 0.0,
        top_rerank: int = TOP_RERANK,
        rerank_batch_size: int = 8,
        *,
        client: Any = None,
        namespace: str = DEFAULT_LOGICAL_NAMESPACE,
        namespace_prefix: str | None = None,
    ) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._embedding_client: Optional[EmbeddingServiceClient] = None
        self.client = client
        self.logical_namespace = namespace
        self.namespace_prefix = (
            namespace_prefix
            or os.getenv("TURBOPUFFER_NAMESPACE_PREFIX")
            or DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX
        )
        self.reranker: CrossEncoderReranker | None = None
        self.embedding_model_name = DEFAULT_EMBEDDING_MODEL_NAME
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
        rerank_threshold: float = 0.0,
        top_rerank: int = TOP_RERANK,
        rerank_batch_size: int = 8,
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
        return retriever

    def _catalog_namespace(self) -> Any:
        name = catalog_namespace(
            self.logical_namespace, namespace_prefix=self.namespace_prefix
        )
        return self.client.namespace(name)

    def _find_catalog_entry(self, table_name: str) -> CorpusCatalogEntry | None:
        catalog = self._catalog_namespace()
        if not catalog.exists():
            return None
        response = catalog.query(
            rank_by=("id", "asc"),
            filters=("id", "Eq", table_name),
            top_k=1,
            include_attributes=True,
        )
        rows = _query_rows(response)
        return CorpusCatalogEntry.from_row(rows[0]) if rows else None

    def _catalog_entry(self, table_name: str) -> CorpusCatalogEntry:
        entry = self._find_catalog_entry(table_name)
        if entry is None:
            raise FileNotFoundError(
                f"No Turbopuffer catalog entry exists for "
                f"{self.logical_namespace}/{table_name}."
            )
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
            self._embedding_client = EmbeddingServiceClient(embedding_service_url)
            logger.info("Using remote embedding service at %s.", embedding_service_url)
            return
        self._load_model(model_name, device, trust_remote_code, max_seq_length)

    def _load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        max_seq_length: Optional[int] = None,
    ) -> None:
        if self._model is not None:
            return
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
        self._model = SentenceTransformer(
            model_name,
            device=model_device,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
        )
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length
        self.embedding_model_name = model_name

    def _load_reranker(
        self,
        reranker_model_name: str,
        device: Optional[str] = None,
        top_rerank: int = TOP_RERANK,
        rerank_threshold: float = 0.0,
        rerank_batch_size: int = 8,
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
        if self._embedding_client is not None:
            return np.ascontiguousarray(
                self._embedding_client.encode_queries([query])[0], dtype=np.float32
            )
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

    @staticmethod
    def _deserialize_row(row: Mapping[str, Any]) -> Document:
        metadata_raw = row.get("metadata_json") or "{}"
        try:
            metadata = json.loads(str(metadata_raw))
        except json.JSONDecodeError as exc:
            raise ValueError("Stored metadata is not valid JSON.") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Stored metadata must decode to an object.")
        return Document(
            id=row["id"],
            title=str(row.get("title") or ""),
            content=str(row.get("content") or ""),
            metadata=metadata,
            document_id=row.get("document_id"),
        )

    @classmethod
    def _row_to_result(cls, row: Mapping[str, Any]) -> RetrievalResult:
        document = cls._deserialize_row(row)
        metadata = dict(document.metadata)
        if document.document_id is not None:
            metadata.setdefault(DOCUMENT_ID_KEY, document.document_id)
        if row.get("_relevance_score") is not None:
            score = row["_relevance_score"]
        elif row.get("$dist") is not None:
            score = row["$dist"]
        else:
            score = 0.0
        return RetrievalResult(
            id=document.id,
            title=document.title,
            content=document.content,
            metadata=metadata,
            score=float(score),
        )

    def get_chunks_table(self, table_name: str) -> Any:
        entry = self._catalog_entry(table_name)
        return self.client.namespace(entry.chunks_namespace)

    def get_documents_table(self, table_name: str) -> Any:
        entry = self._catalog_entry(table_name)
        return self.client.namespace(entry.documents_namespace)

    def count_chunks(self, table_name: str) -> int:
        return self._catalog_entry(table_name).chunk_count

    def get_chunk_by_source_index(
        self, table_name: str, source_index: int
    ) -> Document | None:
        if source_index < 0:
            raise ValueError("source_index must be non-negative.")
        response = self.get_chunks_table(table_name).query(
            rank_by=("source_index", "asc"),
            filters=("source_index", "Eq", source_index),
            top_k=1,
            include_attributes=list(_REMOTE_FIELDS),
        )
        rows = _query_rows(response)
        return self._deserialize_row(rows[0]) if rows else None

    def get_document(self, doc_id: int | str, table_name: str) -> Document | None:
        response = self.get_documents_table(table_name).query(
            rank_by=("id", "asc"),
            filters=("id", "Eq", doc_id),
            top_k=1,
            include_attributes=list(_REMOTE_FIELDS),
        )
        rows = _query_rows(response)
        return self._deserialize_row(rows[0]) if rows else None

    def _require_vectors(self, table_name: str) -> CorpusCatalogEntry:
        entry = self._catalog_entry(table_name)
        if not entry.vector_available:
            raise ValueError(
                f"Turbopuffer corpus {self.logical_namespace}/{table_name} is "
                "lexical-only; dense and hybrid retrieval require vectors."
            )
        return entry

    def retrieve_dense(
        self, query: str, table_name: str, top_k: int = 50
    ) -> list[RetrievalResult]:
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
            include_attributes=list(_REMOTE_FIELDS),
        )
        return [self._row_to_result(row) for row in _query_rows(response)]

    def retrieve_bm25(
        self, query: str, table_name: str, top_k: int = 50
    ) -> list[RetrievalResult]:
        entry = self._catalog_entry(table_name)
        response = self.client.namespace(entry.chunks_namespace).query(
            rank_by=("content", "BM25", query),
            top_k=top_k,
            include_attributes=list(_REMOTE_FIELDS),
        )
        return [self._row_to_result(row) for row in _query_rows(response)]

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
            "include_attributes": list(_REMOTE_FIELDS),
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
        return _multi_query_rows(response)

    def retrieve(
        self,
        query: str,
        table_name: str,
        top_k: int = 50,
        retrieval_mode: Optional[RetrievalMode] = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
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
            return [self._row_to_result(row) for row in rows[:top_k]]
        if self.reranker is None:
            raise RuntimeError("Hybrid reranked retrieval requires a loaded reranker.")
        fetch_limit = max(top_k, self.reranker.top_rerank or top_k)
        rows = self._hybrid_rows(query, table_name, fetch_limit=fetch_limit)
        reranked = self.reranker.rerank(query, rows)
        return [self._row_to_result(row) for row in reranked[:top_k]]

    def scan_chunks(
        self, table_name: str, server_regex: str
    ) -> list[tuple[str, int | str | None, str]]:
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
                    (regex_filter, ("source_index", "Gt", last_source_index)),
                )
            response = namespace.query(
                rank_by=("source_index", "asc"),
                filters=filters,
                top_k=SCAN_PAGE_SIZE,
                include_attributes=[
                    "content",
                    "title",
                    "document_id",
                    "source_index",
                ],
            )
            page = _query_rows(response)
            for row in page:
                rows.append(
                    (
                        str(row.get("content") or ""),
                        row.get("document_id"),
                        str(row.get("title") or ""),
                    )
                )
            if len(page) < SCAN_PAGE_SIZE:
                break
            next_source_index = int(page[-1]["source_index"])
            if last_source_index is not None and next_source_index <= last_source_index:
                raise RuntimeError("Turbopuffer scan pagination did not advance.")
            last_source_index = next_source_index
        return rows


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DEFAULT_EMBEDDING_MODEL_NAME",
    "DEFAULT_LOGICAL_NAMESPACE",
    "DEFAULT_RERANKER_MODEL_NAME",
    "DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX",
    "DEFAULT_TURBOPUFFER_REGION",
    "CorpusCatalogEntry",
    "CrossEncoderReranker",
    "TurbopufferRetriever",
    "catalog_namespace",
    "corpus_namespaces",
    "create_turbopuffer_client",
]
