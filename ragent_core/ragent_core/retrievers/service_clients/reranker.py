from collections.abc import Sequence
from typing import Any

from ragent_core.retrievers.model_contract import (
    RERANK_ROUTE,
    RankResult,
    normalize_rank_results,
)
from ragent_core.retrievers.service_clients.transport import ModelServiceClient


class CrossEncoderServiceClient(ModelServiceClient):
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
        payload = self.post(
            RERANK_ROUTE,
            {"query": str(query), "texts": normalized_texts},
        )
        results = normalize_rank_results(payload, len(normalized_texts))
        return results if top_k is None else results[:top_k]
