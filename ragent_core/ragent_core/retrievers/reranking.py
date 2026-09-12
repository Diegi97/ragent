import logging
from collections.abc import Sequence
from typing import Any, Optional

import torch
from sentence_transformers import CrossEncoder

from ragent_core.retrievers.model_contract import RankResult, normalize_rank_results
from ragent_core.retrievers.settings import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_THRESHOLD,
    DEFAULT_RERANKER_MODEL_NAME,
    TOP_RERANK,
)
from ragent_core.retrievers.storage import RELEVANCE_SCORE_FIELD

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Apply raw-logit CrossEncoder scoring to a retrieval shortlist."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL_NAME,
        device: Optional[str] = None,
        top_rerank: int = TOP_RERANK,
        rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
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
            self._reranker = LocalCrossEncoder(
                CrossEncoder(
                    self.model_name, device=self.device, max_length=self.max_length
                )
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
            row[RELEVANCE_SCORE_FIELD] = score
            selected.append(row)
        return selected


class LocalCrossEncoder:
    """Normalize third-party local model results at the model boundary."""

    def __init__(self, model: CrossEncoder):
        self.model = model

    def rank(self, query: str, texts: Sequence[str], **kwargs: Any) -> list[RankResult]:
        return normalize_rank_results(
            self.model.rank(query, texts, **kwargs), len(texts)
        )
