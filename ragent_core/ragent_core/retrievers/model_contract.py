"""Canonical model wire contract; generates the standalone service copy."""

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, TypedDict

import numpy as np

EMBEDDING_QUERY_ROUTE = "/v1/embeddings/query"
EMBEDDING_DOCUMENTS_ROUTE = "/v1/embeddings/documents"
RERANK_ROUTE = "/v1/rerank"
QUERY_PROMPT_NAME = "web_search_query"


class ModelServiceError(RuntimeError):
    """A model adapter failed to return its declared response contract."""


class RankResult(TypedDict):
    corpus_id: int
    score: float


def normalize_embeddings(payload: Any, count: int) -> np.ndarray:
    try:
        embeddings = np.asarray(payload, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ModelServiceError("Embedding response is not numeric") from exc
    if embeddings.ndim != 2 or embeddings.shape[0] != count or embeddings.shape[1] == 0:
        raise ModelServiceError(
            f"Embedding response shape must have {count} rows and a positive width; got {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ModelServiceError("Embedding response contains non-finite values")
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def normalize_rank_results(payload: Any, count: int) -> list[RankResult]:
    if not isinstance(payload, list):
        raise ModelServiceError("Reranker response must be a list")
    results: list[RankResult] = []
    seen_ids: set[int] = set()
    previous_score = math.inf
    for item in payload:
        if not isinstance(item, Mapping):
            raise ModelServiceError("Each reranker result must be an object")
        corpus_id, score = item.get("corpus_id"), item.get("score")
        if (
            isinstance(corpus_id, bool)
            or not isinstance(corpus_id, Integral)
            or not 0 <= corpus_id < count
        ):
            raise ModelServiceError(f"Invalid reranker corpus_id: {corpus_id!r}")
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(score)
        ):
            raise ModelServiceError("Reranker response contains an invalid score")
        if corpus_id in seen_ids:
            raise ModelServiceError(f"Duplicate reranker corpus_id: {corpus_id}")
        if score > previous_score:
            raise ModelServiceError("Reranker response is not sorted by score")
        seen_ids.add(int(corpus_id))
        previous_score = float(score)
        results.append(RankResult(corpus_id=int(corpus_id), score=float(score)))
    if len(results) != count:
        raise ModelServiceError(
            "Reranker response must contain one result for every input text"
        )
    return results
