import bentoml

from model_services.harrier_service.config import (
    MAX_BATCH_SIZE,
    MAX_LATENCY_MS,
    TIMEOUT_SECONDS,
    service_image,
)
from model_services.harrier_service.model import HarrierEncoder
from model_services.model_contract import (
    EMBEDDING_DOCUMENTS_ROUTE,
    EMBEDDING_QUERY_ROUTE,
)


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
        route=EMBEDDING_QUERY_ROUTE,
        batchable=True,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode_queries(texts)

    @bentoml.api(
        route=EMBEDDING_DOCUMENTS_ROUTE,
        batchable=True,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encoder.encode_documents(texts)
