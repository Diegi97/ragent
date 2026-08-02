import os
from typing import Any

import bentoml
import torch
from sentence_transformers import CrossEncoder

DEFAULT_MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_device(configured: str) -> str:
    if configured != "auto":
        if configured not in {"cpu", "cuda", "mps"}:
            raise ValueError("MXBAI_DEVICE must be one of auto, cpu, cuda, or mps")
        return configured
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MODEL_ID = os.getenv("MXBAI_MODEL_ID", DEFAULT_MODEL_ID)
DEVICE = _resolve_device(os.getenv("MXBAI_DEVICE", "auto"))
MAX_LENGTH = _env_int("MXBAI_MAX_LENGTH", 512)
INFERENCE_BATCH_SIZE = _env_int("MXBAI_INFERENCE_BATCH_SIZE", 8)
TIMEOUT_SECONDS = _env_int("MXBAI_TIMEOUT_SECONDS", 60)

service_image = bentoml.images.Image(python_version="3.12").python_packages(
    "sentence-transformers==5.6.0",
    "torch==2.11.0",
    "transformers==5.12.1",
)


class MxbaiRanker:
    """CrossEncoder adapter that returns sorted raw logits."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model or CrossEncoder(
            MODEL_ID,
            device=DEVICE,
            max_length=MAX_LENGTH,
        )

    def rank(self, query: str, texts: list[str]) -> list[dict[str, int | float]]:
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
        normalized_results = [
            {
                "corpus_id": int(result["corpus_id"]),
                "score": float(result["score"]),
            }
            for result in results
        ]
        return sorted(
            normalized_results,
            key=lambda result: result["score"],
            reverse=True,
        )


@bentoml.service(
    name="ragent-mxbai-reranker",
    image=service_image,
    workers=1,
    traffic={"timeout": TIMEOUT_SECONDS},
)
class MxbaiRerankerService:
    def __init__(self) -> None:
        self.ranker = MxbaiRanker()

    @bentoml.api(route="/v1/rerank")
    def rerank(
        self,
        query: str,
        texts: list[str],
    ) -> list[dict[str, int | float]]:
        return self.ranker.rank(query, texts)
