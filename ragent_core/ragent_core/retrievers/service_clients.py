import math
import os
from collections.abc import Sequence
from typing import Any, TypedDict

import httpx
import numpy as np
from httpx_retries import Retry, retry_request

RETRYABLE_STATUS_CODES = frozenset({408, 429, 502, 503, 504})
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_SECONDS = 1.0
DEFAULT_RETRY_MAX_SECONDS = 30.0


def _service_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _service_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


class ModelServiceError(RuntimeError):
    """Raised when a remote model service returns an invalid response."""


class RankResult(TypedDict):
    corpus_id: int
    score: float


class _ModelServiceClient:
    def __init__(
        self,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Model service base_url cannot be empty")
        timeout = _service_float(
            "RAGENT_MODEL_SERVICE_TIMEOUT", DEFAULT_TIMEOUT_SECONDS
        )
        connect_timeout = _service_float(
            "RAGENT_MODEL_SERVICE_CONNECT_TIMEOUT",
            DEFAULT_CONNECT_TIMEOUT_SECONDS,
        )
        max_retries = _service_int(
            "RAGENT_MODEL_SERVICE_MAX_RETRIES", DEFAULT_MAX_RETRIES
        )
        retry_base_seconds = _service_float(
            "RAGENT_MODEL_SERVICE_RETRY_BASE_SECONDS",
            DEFAULT_RETRY_BASE_SECONDS,
        )
        retry_max_seconds = _service_float(
            "RAGENT_MODEL_SERVICE_RETRY_MAX_SECONDS",
            DEFAULT_RETRY_MAX_SECONDS,
        )
        if timeout <= 0:
            raise ValueError("RAGENT_MODEL_SERVICE_TIMEOUT must be positive")
        if connect_timeout <= 0:
            raise ValueError("RAGENT_MODEL_SERVICE_CONNECT_TIMEOUT must be positive")
        if max_retries < 0:
            raise ValueError("RAGENT_MODEL_SERVICE_MAX_RETRIES cannot be negative")
        if retry_base_seconds < 0:
            raise ValueError(
                "RAGENT_MODEL_SERVICE_RETRY_BASE_SECONDS cannot be negative"
            )
        if retry_max_seconds < retry_base_seconds:
            raise ValueError(
                "RAGENT_MODEL_SERVICE_RETRY_MAX_SECONDS must be >= "
                "RAGENT_MODEL_SERVICE_RETRY_BASE_SECONDS"
            )
        self.base_url = base_url.rstrip("/")
        self._retry = Retry(
            total=max_retries,
            max_backoff_wait=retry_max_seconds,
            backoff_factor=retry_base_seconds,
            respect_retry_after_header=True,
            allowed_methods={"POST"},
            status_forcelist=RETRYABLE_STATUS_CODES,
            backoff_jitter=1.0,
        )
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "_ModelServiceClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = retry_request(
            self._client,
            "POST",
            f"{self.base_url}{path}",
            retry=self._retry,
            json=payload,
        )
        try:
            response.raise_for_status()
            try:
                return response.json()
            except ValueError as exc:
                raise ModelServiceError(
                    f"Model service returned non-JSON data for {path}"
                ) from exc
        finally:
            response.close()


class EmbeddingServiceClient(_ModelServiceClient):
    def _encode(self, path: str, texts: Sequence[str]) -> np.ndarray:
        normalized_texts = [str(text) for text in texts]
        if not normalized_texts:
            return np.empty((0, 0), dtype=np.float32)
        payload = self._post(path, {"texts": normalized_texts})
        try:
            embeddings = np.asarray(payload, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ModelServiceError("Embedding response is not numeric") from exc
        if embeddings.ndim != 2 or embeddings.shape[0] != len(normalized_texts):
            raise ModelServiceError(
                "Embedding response shape does not match the requested texts: "
                f"expected {len(normalized_texts)} rows, got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ModelServiceError("Embedding response contains non-finite values")
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("/v1/embeddings/query", texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("/v1/embeddings/documents", texts)


class CrossEncoderServiceClient(_ModelServiceClient):
    def rank(
        self,
        query: str,
        texts: Sequence[str],
        *,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> list[RankResult]:
        normalized_texts = [str(text) for text in texts]
        if not normalized_texts:
            return []
        payload = self._post(
            "/v1/rerank",
            {"query": str(query), "texts": normalized_texts},
        )
        if not isinstance(payload, list):
            raise ModelServiceError("Reranker response must be a list")

        results: list[RankResult] = []
        seen_ids: set[int] = set()
        previous_score = math.inf
        for item in payload:
            if not isinstance(item, dict):
                raise ModelServiceError("Each reranker result must be an object")
            try:
                corpus_id = int(item["corpus_id"])
                score = float(item["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelServiceError("Invalid reranker result") from exc
            if corpus_id < 0 or corpus_id >= len(normalized_texts):
                raise ModelServiceError(f"Invalid reranker corpus_id: {corpus_id}")
            if corpus_id in seen_ids:
                raise ModelServiceError(f"Duplicate reranker corpus_id: {corpus_id}")
            if not math.isfinite(score):
                raise ModelServiceError("Reranker response contains a non-finite score")
            if score > previous_score:
                raise ModelServiceError("Reranker response is not sorted by score")
            seen_ids.add(corpus_id)
            previous_score = score
            results.append({"corpus_id": corpus_id, "score": score})

        if len(results) != len(normalized_texts):
            raise ModelServiceError(
                "Reranker response must contain one result for every input text"
            )
        return results if top_k is None else results[:top_k]
