import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)

if TYPE_CHECKING:
    from ragent_core.retrievers.retriever import TurbopufferRetriever


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


async def chat_completion(
    messages: Sequence[dict[str, str]],
    config: RetrievalQueriesConfig,
) -> LLMCompletion:
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://api.fireworks.ai/inference/v1",
        ),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "5")),
    )
    async with client:
        response = await client.chat.completions.create(
            model=config.generator_model,
            messages=list(messages),
        )
    usage = response.usage
    return LLMCompletion(
        content=response.choices[0].message.content if response.choices else "",
        model=response.model or config.generator_model,
        prompt_tokens=usage.prompt_tokens if usage is not None else None,
        completion_tokens=usage.completion_tokens if usage is not None else None,
        total_tokens=usage.total_tokens if usage is not None else None,
    )


_retriever_lock = threading.Lock()


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
    return os.getenv("TURBOPUFFER_NAMESPACE_PREFIX", "ragent")


def count_chunks(config: RetrievalQueriesConfig) -> int:
    return _cached_catalog_retriever(
        config.logical_namespace,
        _namespace_prefix(),
    ).count_chunks(config.table_name)


def chunk_by_source_index(
    config: RetrievalQueriesConfig, source_index: int
) -> dict[str, Any] | None:
    document = _cached_catalog_retriever(
        config.logical_namespace,
        _namespace_prefix(),
    ).get_chunk_by_source_index(config.table_name, source_index)
    return document.to_dict() if document is not None else None


def load_retriever(config: RetrievalQueriesConfig) -> "TurbopufferRetriever":
    embedding_service_url = os.getenv("RAGENT_EMBEDDING_SERVICE_URL")
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
