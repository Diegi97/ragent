from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

DocumentId: TypeAlias = int | str


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
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class EvaluationSummary:
    output_directory: Path
    summary_path: Path
    details_path: Path
    total_queries: int
    successful_queries: int
    failed_queries: int
    metrics: dict[str, Any]

    @property
    def coverage(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries
