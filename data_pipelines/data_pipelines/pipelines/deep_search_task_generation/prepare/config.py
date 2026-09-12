from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from data_pipelines.pipelines.deep_search_task_generation.config import PROJECT_ROOT
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.config import (
    DEFAULT_RETRIEVER_WORKER_PORT,
)

DEFAULT_ENTITY_MODEL_ID = "accounts/fireworks/models/deepseek-v4-flash-0731"


class DeepSearchTaskGenerationConfig(BaseModel):
    """Reproducible parameters persisted by the prepare phase."""

    data_source: str = Field(min_length=1)
    output_root: Path = PROJECT_ROOT / "data/deep_search_task_generation"
    entities_file: Path | None = None
    num_entities: int = Field(default=100, ge=0)
    entity_model_id: str = DEFAULT_ENTITY_MODEL_ID
    seed: int = 42
    sample_size: int = Field(default=4, ge=1)
    num_chunks_per_entity: int = Field(default=100, ge=1)
    fact_extraction_chunks_per_request: int = Field(default=5, ge=1)
    llm_concurrency: int = Field(default=25, ge=1)
    retriever_worker_port: int = Field(
        default=DEFAULT_RETRIEVER_WORKER_PORT,
        ge=1,
        le=65535,
    )
    upload_timeout: float = Field(default=600.0, gt=0.0)

    @field_validator("output_root", mode="after")
    @classmethod
    def resolve_paths(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("entities_file", mode="after")
    @classmethod
    def resolve_optional_paths(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None
