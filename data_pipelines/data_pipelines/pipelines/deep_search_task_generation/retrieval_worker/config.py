from typing import Any

from pydantic import BaseModel, Field

from ragent_core.retrievers.settings import DEFAULT_LOGICAL_NAMESPACE

DEFAULT_RETRIEVER_WORKER_PORT = 8765


class RetrieverWorkerConfig(BaseModel):
    retriever_namespace: str = Field(default=DEFAULT_LOGICAL_NAMESPACE, min_length=1)
    retriever_device: str | None = None
    rerank_threshold: float = Field(default=3.0, ge=0.0, allow_inf_nan=False)
    port: int = Field(default=DEFAULT_RETRIEVER_WORKER_PORT, ge=1, le=65535)

    def public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
