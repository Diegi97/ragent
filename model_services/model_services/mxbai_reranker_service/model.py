from typing import Any

import torch
from sentence_transformers import CrossEncoder

from model_services.model_contract import RankResult, normalize_rank_results
from model_services.mxbai_reranker_service.config import (
    DEVICE,
    INFERENCE_BATCH_SIZE,
    MAX_LENGTH,
    MODEL_ID,
)


class MxbaiRanker:
    """CrossEncoder adapter that returns sorted raw logits."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model or CrossEncoder(
            MODEL_ID,
            device=DEVICE,
            max_length=MAX_LENGTH,
        )

    def rank(self, query: str, texts: list[str]) -> list[RankResult]:
        if not texts:
            return []
        with torch.inference_mode():
            results = self.model.rank(
                query,
                texts,
                top_k=len(texts),
                return_documents=False,
                batch_size=INFERENCE_BATCH_SIZE,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
            )
        sorted_results = sorted(
            results, key=lambda result: result["score"], reverse=True
        )
        return normalize_rank_results(sorted_results, len(texts))
