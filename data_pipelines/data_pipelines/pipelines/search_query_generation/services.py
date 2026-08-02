from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import lancedb
from openai import AsyncOpenAI

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)

if TYPE_CHECKING:
    from ragent_core.retrievers.retriever import LanceDBRetriever


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


def chunks_table(config: RetrievalQueriesConfig) -> Any:
    uri = f"{str(config.lancedb_db_uri).rstrip('/')}/{config.retriever_namespace}"
    return lancedb.connect(uri=uri).open_table(f"{config.table_name}_chunks")


_retriever_lock = threading.Lock()


@lru_cache(maxsize=8)
def _cached_retriever(
    namespace: str,
    embedding_service_url: str,
    lancedb_db_uri: str,
) -> LanceDBRetriever:
    from ragent_core.retrievers.retriever import LanceDBRetriever

    # ragent_core resolves its database URI from this environment variable.
    os.environ["LANCEDB_DB_URI"] = lancedb_db_uri
    return LanceDBRetriever.load_index(
        namespace=namespace,
        reranker_model_name=None,
        embedding_service_url=embedding_service_url,
    )


def load_retriever(config: RetrievalQueriesConfig) -> LanceDBRetriever:
    embedding_service_url = os.getenv("RAGENT_EMBEDDING_SERVICE_URL")
    if not embedding_service_url:
        raise RuntimeError(
            "Set RAGENT_EMBEDDING_SERVICE_URL before running search query generation."
        )
    key = (
        config.retriever_namespace,
        embedding_service_url,
        str(config.lancedb_db_uri),
    )
    with _retriever_lock:
        return _cached_retriever(*key)
