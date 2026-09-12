from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_pipelines.pipelines.retrieval_evaluation.contracts import EvaluationMetrics
from data_pipelines.pipelines.search_query_generation.metadata import QueryRunMetadata
from ragent_core.retrievers.document import DocumentId


@dataclass(frozen=True)
class QueryRecord:
    index: int
    line_number: int
    query: str
    positive_chunk_id: DocumentId
    positive_document_id: DocumentId
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetContext:
    input_directory: Path
    queries_path: Path
    metadata_path: Path
    table_name: str
    logical_namespace: str
    source_metadata: QueryRunMetadata

    def to_provenance(self) -> dict[str, Any]:
        metadata = self.source_metadata
        source_config = metadata.config
        return {
            "input_directory": str(self.input_directory),
            "queries_path": str(self.queries_path),
            "metadata_path": str(self.metadata_path),
            "batch_timestamp": metadata.batch_timestamp,
            "prefect_flow_run_id": metadata.prefect_flow_run_id,
            "generator_model": source_config.generator_model,
            "requested_records": metadata.requested_records,
            "trainable_records": metadata.trainable_records,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    output_directory: Path
    summary_path: Path
    details_path: Path
    total_queries: int
    successful_queries: int
    failed_queries: int
    metrics: EvaluationMetrics

    @property
    def coverage(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries
