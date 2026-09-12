from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)


@dataclass
class Concept:
    """Concept extracted from source documents for QA generation."""

    name: str
    data_source: str
    doc_id: int
    importance: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_name(self) -> str:
        return self.name.strip().lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_source": self.data_source,
            "doc_id": self.doc_id,
            "importance": self.importance,
            "info": self.info,
        }


@dataclass(frozen=True)
class PreparePaths:
    directory: Path
    fact_responses: Path
    retrieval_debug_directory: Path
    entities: Path
    fact_requests: Path
    failures: Path
    metadata: Path
    lock: Path

    @classmethod
    def in_directory(cls, directory: Path) -> "PreparePaths":
        return cls(
            directory=directory,
            fact_responses=directory / "fact_responses.jsonl",
            retrieval_debug_directory=directory / "retrieval_debug",
            entities=directory / "entities.jsonl",
            fact_requests=directory / "fact_requests.jsonl",
            failures=directory / "failures.jsonl",
            metadata=directory / "prepare_metadata.json",
            lock=directory / ".records.lock",
        )


class PrepareStatus(StrEnum):
    FAILED = "failed"
    EMPTY = "empty"
    PREPARED = "prepared"
    UPLOADED = "uploaded"
    UPLOAD_FAILED = "upload_failed"


class PrepareFireworksMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_dataset_name: str | None = None
    upload_payload: dict[str, Any] = Field(default_factory=dict)


class PrepareRunMetadata(BaseModel):
    """Prepare artifact contract; optional diagnostics preserve older runs."""

    model_config = ConfigDict(extra="allow")

    config: DeepSearchTaskGenerationConfig
    status: PrepareStatus | None = None
    prefect_flow_run_id: str | None = None
    phoenix_project: str | None = None
    retriever_worker: dict[str, Any] = Field(default_factory=dict)
    data_source: str | None = None
    table_name: str | None = None
    requested_entities: int | None = Field(default=None, ge=0)
    retained_entities: int | None = Field(default=None, ge=0)
    fact_request_count: int | None = Field(default=None, ge=0)
    failure_count: int | None = Field(default=None, ge=0)
    crashed_entities: int | None = Field(default=None, ge=0)
    fireworks: PrepareFireworksMetadata = Field(
        default_factory=PrepareFireworksMetadata
    )
    paths: dict[str, str] = Field(default_factory=dict)


class PrepareStage(StrEnum):
    ENTITY_EXTRACTION = "entity_extraction"
    FACT_REQUEST_PREPARATION = "fact_request_preparation"
    DATASET_UPLOAD = "dataset_upload"


class PrepareFailure(TypedDict):
    stage: PrepareStage
    error: str
    entity: NotRequired[str]
    doc_id: NotRequired[int]
    crashed: NotRequired[bool]
