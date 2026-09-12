from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from data_pipelines.artifacts.retrieval_queries import (
    RetrievalChunk,
    TrainingRecord,
)
from ragent_core.retrievers.document import DocumentId


class QueryStatus(StrEnum):
    SAMPLED = "sampled"
    GENERATED = "generated"
    MINED = "mined"
    READY = "ready"
    FILTERED = "filtered"
    FAILED = "failed"


class FilterReason(StrEnum):
    ROUND_TRIP_MISS = "round_trip_miss"
    CONTRASTIVE_REJECTION = "contrastive_rejection"


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    doc_id: DocumentId | None
    positive: RetrievalChunk | None = None
    hard_negatives: tuple[RetrievalChunk, ...] = field(default_factory=tuple)
    candidates: tuple[RetrievalChunk, ...] = field(default_factory=tuple)
    status: QueryStatus = QueryStatus.SAMPLED
    failure_reason: str | None = None
    reason_code: FilterReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, status: QueryStatus, **metadata: Any) -> "RetrievalQuery":
        return replace(
            self,
            status=status,
            failure_reason=None,
            reason_code=None,
            metadata={**self.metadata, **metadata},
        )

    def failed(
        self,
        status: QueryStatus,
        reason: str,
        reason_code: FilterReason | None = None,
        **metadata: Any,
    ) -> "RetrievalQuery":
        return replace(
            self,
            status=status,
            failure_reason=reason,
            reason_code=reason_code,
            metadata={**self.metadata, **metadata},
        )

    def is_trainable(self) -> bool:
        return (
            self.status is QueryStatus.READY
            and bool(self.query)
            and self.positive is not None
        )

    @staticmethod
    def resolve_hard_negatives(
        selected_ids: list[str], candidates: tuple[RetrievalChunk, ...]
    ) -> tuple[RetrievalChunk, ...]:
        candidate_by_id = {str(candidate.id): candidate for candidate in candidates}
        return tuple(
            candidate_by_id[chunk_id]
            for chunk_id in selected_ids
            if chunk_id in candidate_by_id
        )

    def to_training_record(self, hard_negatives_per_query: int) -> dict[str, Any]:
        if self.positive is None:
            raise ValueError(
                "Cannot serialize a training record without a positive chunk."
            )
        hard_negatives = self.hard_negatives[:hard_negatives_per_query]
        return TrainingRecord(
            query=self.query,
            positive=self.positive,
            hard_negatives=hard_negatives,
            metadata={**self.metadata, "doc_id": self.doc_id, "status": self.status},
        ).to_dict()

    def to_failure_record(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "positive": self.positive.to_dict() if self.positive is not None else None,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "reason_code": self.reason_code,
            "metadata": self.metadata,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize the complete inspectable object state for Phoenix."""
        return self._to_dict(include_document_text=True)

    def _to_dict(self, include_document_text: bool) -> dict[str, Any]:
        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "positive": (
                self.positive.to_dict(include_text=include_document_text)
                if self.positive is not None
                else None
            ),
            "hard_negatives": [
                chunk.to_dict(include_text=include_document_text)
                for chunk in self.hard_negatives
            ],
            "candidates": [
                chunk.to_dict(include_text=include_document_text)
                for chunk in self.candidates
            ],
            "status": self.status,
            "failure_reason": self.failure_reason,
            "reason_code": self.reason_code,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class OutputPaths:
    output_directory: Path
    output_path: Path
    failures_path: Path
    metadata_path: Path
    lock_path: Path


@dataclass(frozen=True)
class ObjectRunSummary:
    object_id: str
    sample_index: int
    row_index: int
    status: QueryStatus
    phoenix_trace_id: str
    crashed: bool = False
    error: str | None = None
    record_path: str | None = None


class ObjectPipelineError(RuntimeError):
    def __init__(self, summary: ObjectRunSummary):
        self.summary = summary
        super().__init__(
            summary.error or f"Object pipeline failed: {summary.object_id}"
        )
