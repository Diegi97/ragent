from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from model_services.harrier_service.config import (
    DEVICE,
    INFERENCE_BATCH_SIZE,
    MAX_SEQ_LENGTH,
    MODEL_ID,
)
from model_services.model_contract import QUERY_PROMPT_NAME, normalize_embeddings


class HarrierEncoder:
    """Model adapter that keeps query and document prompting explicit."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model or SentenceTransformer(
            MODEL_ID,
            device=DEVICE,
            trust_remote_code=True,
            model_kwargs={"dtype": "auto"},
        )
        self.model.max_seq_length = MAX_SEQ_LENGTH

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, QUERY_PROMPT_NAME)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, None)

    def _encode(self, texts: list[str], prompt_name: str | None) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {
            "batch_size": INFERENCE_BATCH_SIZE,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        with torch.inference_mode():
            embeddings = self.model.encode(texts, **kwargs)
        return normalize_embeddings(embeddings, len(texts)).tolist()
