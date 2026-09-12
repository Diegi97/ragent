from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from data_pipelines.pipelines.search_query_generation.config import (
    RetrievalQueriesConfig,
)


class QueryCrashMetadata(BaseModel):
    object_id: str
    sample_index: int
    row_index: int
    error: str | None = None


class QueryRunMetadata(BaseModel):
    """Persisted query-run contract, including optional historical diagnostics."""

    model_config = ConfigDict(extra="allow")

    config: RetrievalQueriesConfig
    requested_records: int | None = Field(default=None, ge=0)
    reserved_records: int | None = Field(default=None, ge=0)
    total_records: int | None = Field(default=None, ge=0)
    trainable_records: int | None = Field(default=None, ge=0)
    failure_records: int | None = Field(default=None, ge=0)
    crashed_records: int | None = Field(default=None, ge=0)
    crashes: list[QueryCrashMetadata] = Field(default_factory=list)
    prefect_flow_run_id: str | None = None
    batch_timestamp: str | None = None
    phoenix_project: str | None = None
    output_directory: str | None = None
    output_path: str | None = None
    failures_path: str | None = None

    @field_validator("config", mode="before")
    @classmethod
    def require_persisted_namespace(cls, value: Any) -> Any:
        if isinstance(value, RetrievalQueriesConfig):
            namespace = value.logical_namespace
        elif isinstance(value, dict):
            namespace = value.get("logical_namespace")
        else:
            namespace = None
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError(
                "metadata.json config.logical_namespace must be a non-empty string."
            )
        return value
