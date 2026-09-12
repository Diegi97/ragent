import os
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.config import (
    RetrieverWorkerConfig,
)
from ragent_core.retrievers.document import RetrievalResultData

# Version 2 moves the pickled request/response classes into this contract module.
PROTOCOL_VERSION = 2
RETRIEVER_WORKER_HOST = "127.0.0.1"
RETRIEVER_AUTHKEY_ENV = "RAGENT_RETRIEVER_AUTHKEY"


@dataclass(frozen=True)
class RetrievalRequest:
    protocol_version: int
    client_id: str
    request_id: str
    query: str
    table_name: str
    top_k: int


@dataclass(frozen=True)
class RetrievalError:
    type: str
    message: str


@dataclass(frozen=True)
class RetrievalResponse:
    protocol_version: int
    client_id: str
    request_id: str
    results: tuple[RetrievalResultData, ...]
    error: RetrievalError | None


class RetrieverWorkerRemoteError(RuntimeError):
    def __init__(self, error: RetrievalError) -> None:
        super().__init__(f"Retriever worker failed: {error.type}: {error.message}")
        self.error_type = error.type


def authkey_from_environment() -> bytes:
    value = os.getenv(RETRIEVER_AUTHKEY_ENV, "")
    if not value:
        raise ValueError(f"{RETRIEVER_AUTHKEY_ENV} is not set")
    return value.encode("utf-8")


class WorkerStatus(StrEnum):
    READY = "ready"
    STOPPING = "stopping"


class WorkerHealth(BaseModel):
    protocol_version: int = Field(strict=True)
    status: WorkerStatus
    host: str
    port: int = Field(ge=1, le=65535)
    pid: int = Field(gt=0)
    uptime_seconds: float = Field(ge=0, allow_inf_nan=False)
    queued_requests: int = Field(ge=0)
    active_request_id: str | None
    connected_clients: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    config: RetrieverWorkerConfig

    @field_validator("protocol_version")
    @classmethod
    def require_protocol(cls, value: int) -> int:
        if value != PROTOCOL_VERSION:
            raise ValueError(
                f"Retriever protocol mismatch: client={PROTOCOL_VERSION}, worker={value}"
            )
        return value
