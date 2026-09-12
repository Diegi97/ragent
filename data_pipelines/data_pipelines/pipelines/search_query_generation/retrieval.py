import os
import threading
from functools import lru_cache
from typing import TYPE_CHECKING

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)
from ragent_core.retrievers.document import Document
from ragent_core.retrievers.settings import (
    DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX,
    EMBEDDING_SERVICE_URL_ENV,
    NAMESPACE_PREFIX_ENV,
)

if TYPE_CHECKING:
    from ragent_core.retrievers.retriever import TurbopufferRetriever


_retriever_lock = threading.Lock()


def load_retriever(config: RetrievalQueriesConfig) -> "TurbopufferRetriever":
    embedding_service_url = os.getenv(EMBEDDING_SERVICE_URL_ENV)
    if not embedding_service_url:
        raise RuntimeError(
            "Set RAGENT_EMBEDDING_SERVICE_URL before running search query generation."
        )
    key = (
        config.logical_namespace,
        embedding_service_url,
        _namespace_prefix(),
    )
    with _retriever_lock:
        return _cached_retriever(*key)


def count_chunks(config: RetrievalQueriesConfig) -> int:
    return _cached_catalog_retriever(
        config.logical_namespace,
        _namespace_prefix(),
    ).count_chunks(config.table_name)


def chunk_by_source_index(
    config: RetrievalQueriesConfig, source_index: int
) -> Document | None:
    document = _cached_catalog_retriever(
        config.logical_namespace,
        _namespace_prefix(),
    ).get_chunk_by_source_index(config.table_name, source_index)
    return document


@lru_cache(maxsize=8)
def _cached_retriever(
    namespace: str,
    embedding_service_url: str,
    namespace_prefix: str,
) -> "TurbopufferRetriever":
    from ragent_core.retrievers.retriever import TurbopufferRetriever

    return TurbopufferRetriever.load_index(
        namespace=namespace,
        namespace_prefix=namespace_prefix,
        reranker_model_name=None,
        embedding_service_url=embedding_service_url,
    )


@lru_cache(maxsize=8)
def _cached_catalog_retriever(
    namespace: str,
    namespace_prefix: str,
) -> "TurbopufferRetriever":
    from ragent_core.retrievers.retriever import TurbopufferRetriever

    return TurbopufferRetriever.load_index(
        namespace=namespace,
        namespace_prefix=namespace_prefix,
        reranker_model_name=None,
        load_embedding_backend=False,
    )


def _namespace_prefix() -> str:
    return os.getenv(NAMESPACE_PREFIX_ENV, DEFAULT_TURBOPUFFER_NAMESPACE_PREFIX)
