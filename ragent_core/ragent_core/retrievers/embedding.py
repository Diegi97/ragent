import importlib
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ragent_core.retrievers.model_contract import (
    QUERY_PROMPT_NAME,
    normalize_embeddings,
)
from ragent_core.retrievers.service_clients.embedding import EmbeddingServiceClient
from ragent_core.retrievers.settings import DEFAULT_EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        *,
        model: Any = None,
        client: EmbeddingServiceClient | None = None,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.client = client

    @classmethod
    def load(
        cls,
        model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
        device: str | None = None,
        trust_remote_code: bool = True,
        max_seq_length: int | None = None,
        embedding_service_url: str | None = None,
    ) -> "EmbeddingBackend":
        if embedding_service_url is not None:
            return cls(model_name, client=EmbeddingServiceClient(embedding_service_url))
        model_kwargs: dict[str, Any] = {"dtype": "auto"}
        if (
            device in (None, "cuda")
            and torch.cuda.is_available()
            and _flash_attention_available()
        ):
            device = "cuda"
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            }
        logger.info("Loading embedding model '%s'", model_name)
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
        )
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
        return cls(model_name, model=model)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.client is not None:
            return self.client.encode_queries(texts)
        return self._encode(texts, batch_size=1, prompt_name=QUERY_PROMPT_NAME)

    def encode_documents(self, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        if self.client is not None:
            return self.client.encode_documents(texts)
        return self._encode(texts, batch_size=batch_size)

    def _encode(
        self, texts: Sequence[str], *, batch_size: int, prompt_name: str | None = None
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")
        options: dict[str, Any] = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        if prompt_name is not None:
            options["prompt_name"] = prompt_name
        with torch.inference_mode():
            payload = self.model.encode(list(texts), **options)
        return normalize_embeddings(payload, len(texts))


def _flash_attention_available() -> bool:
    try:
        importlib.import_module("flash_attn")
    except ImportError:
        return False
    return True
