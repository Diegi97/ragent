import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from ragent_core.retrievers.retriever import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_RERANKER_MODEL_NAME,
)


def _service_url(environment_variable: str) -> str | None:
    value = os.getenv(environment_variable)
    return value.strip() if value and value.strip() else None


class SearchType(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RetrievalEvaluationConfig(BaseModel):
    """Configuration for one retrieval evaluation run."""

    input_directory: Path
    search_type: SearchType = SearchType.HYBRID
    top_k: int = Field(default=50, ge=1)
    cutoffs: tuple[int, ...] = (1, 3, 5, 10, 20, 50)

    embedding_model: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_service_url: str | None = Field(
        default_factory=lambda: _service_url("RAGENT_EMBEDDING_SERVICE_URL")
    )
    device: str | None = None
    max_seq_length: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = True

    reranker: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL_NAME
    reranker_service_url: str | None = Field(
        default_factory=lambda: _service_url("RAGENT_RERANKER_SERVICE_URL")
    )
    reranker_candidate_k: int = Field(default=50, ge=1)
    reranker_threshold: float = 0.0
    reranker_batch_size: int = Field(default=8, ge=1)

    @field_validator("input_directory", mode="after")
    @classmethod
    def resolve_input_directory(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("embedding_model", "reranker_model", mode="after")
    @classmethod
    def require_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Model names cannot be empty.")
        return value

    @field_validator(
        "embedding_service_url",
        "reranker_service_url",
        "device",
        mode="after",
    )
    @classmethod
    def normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("cutoffs", mode="after")
    @classmethod
    def validate_cutoffs(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("At least one metric cutoff is required.")
        if any(cutoff <= 0 for cutoff in value):
            raise ValueError("Metric cutoffs must be positive integers.")
        if any(left >= right for left, right in zip(value, value[1:])):
            raise ValueError("Metric cutoffs must be unique and strictly increasing.")
        return value

    @model_validator(mode="after")
    def validate_retrieval_configuration(self) -> "RetrievalEvaluationConfig":
        if max(self.cutoffs) > self.top_k:
            raise ValueError("The largest metric cutoff cannot exceed top_k.")
        if self.reranker and self.search_type is not SearchType.HYBRID:
            raise ValueError("Reranking is supported only for hybrid search.")
        if self.reranker and self.reranker_candidate_k < self.top_k:
            raise ValueError(
                "reranker_candidate_k must be greater than or equal to top_k."
            )
        return self
