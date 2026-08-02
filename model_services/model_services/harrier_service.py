import os
from typing import Any

import bentoml
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_ID = "microsoft/harrier-oss-v1-0.6b"
QUERY_PROMPT_NAME = "web_search_query"


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        if configured not in {"cpu", "cuda", "mps"}:
            raise ValueError("HARRIER_DEVICE must be one of auto, cpu, cuda, or mps")
        return configured
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MODEL_ID = os.getenv("HARRIER_MODEL_ID", DEFAULT_MODEL_ID)
DEVICE = _resolve_device(os.getenv("HARRIER_DEVICE", "auto"))
MAX_SEQ_LENGTH = _env_int("HARRIER_MAX_SEQ_LENGTH", 512)
INFERENCE_BATCH_SIZE = _env_int("HARRIER_INFERENCE_BATCH_SIZE", 8)
MAX_BATCH_SIZE = _env_int("HARRIER_MAX_BATCH_SIZE", 16)
MAX_LATENCY_MS = _env_int("HARRIER_MAX_LATENCY_MS", 25)
TIMEOUT_SECONDS = _env_int("HARRIER_TIMEOUT_SECONDS", 60)

service_image = bentoml.images.Image(python_version="3.12").python_packages(
    "numpy==2.3.5",
    "sentence-transformers==5.6.0",
    "torch==2.11.0",
    "transformers==5.12.1",
)


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
        return np.ascontiguousarray(embeddings, dtype=np.float32).tolist()

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, QUERY_PROMPT_NAME)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, None)


@bentoml.service(
    name="ragent-harrier-embeddings",
    image=service_image,
    workers=1,
    traffic={"timeout": TIMEOUT_SECONDS},
)
class HarrierEmbeddingService:
    def __init__(self) -> None:
        self.encoder = HarrierEncoder()

    @bentoml.api(
        route="/v1/embeddings/query",
        batchable=True,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode_queries(texts)

    @bentoml.api(
        route="/v1/embeddings/documents",
        batchable=True,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode_documents(texts)
