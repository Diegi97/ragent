from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeAlias

DocumentId: TypeAlias = int | str


@dataclass(frozen=True)
class RetrievalChunk:
    id: DocumentId
    title: str = ""
    text: str = ""
    document_id: DocumentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    sources: tuple[str, ...] = field(default_factory=tuple)
    source_ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        value = {
            "id": self.id,
            "title": self.title,
            "document_id": self.document_id,
            "metadata": self.metadata,
            "score": self.score,
            "sources": list(self.sources),
            "source_ranks": self.source_ranks,
        }
        if include_text:
            value["text"] = self.text
        return value


@dataclass(frozen=True)
class RetrievalQuery:
    query: str
    doc_id: DocumentId | None
    positive: RetrievalChunk | None = None
    hard_negatives: tuple[RetrievalChunk, ...] = field(default_factory=tuple)
    candidates: tuple[RetrievalChunk, ...] = field(default_factory=tuple)
    status: str = "sampled"
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, status: str, **metadata: Any) -> "RetrievalQuery":
        return replace(
            self,
            status=status,
            failure_reason=None,
            metadata={**self.metadata, **metadata},
        )

    def failed(
        self,
        status: str,
        reason: str,
        **metadata: Any,
    ) -> "RetrievalQuery":
        return replace(
            self,
            status=status,
            failure_reason=reason,
            metadata={**self.metadata, **metadata},
        )

    def is_trainable(self) -> bool:
        return self.status == "ready" and bool(self.query) and self.positive is not None

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
            "metadata": self.metadata,
        }

    def to_training_record(self, hard_negatives_per_query: int) -> dict[str, Any]:
        if self.positive is None:
            raise ValueError(
                "Cannot serialize a training record without a positive chunk."
            )
        hard_negatives = self.hard_negatives[:hard_negatives_per_query]
        return {
            "query": self.query,
            "positive": self.positive.to_dict(),
            "hard_negatives": [chunk.to_dict() for chunk in hard_negatives],
            "metadata": {
                **self.metadata,
                "doc_id": self.doc_id,
                "status": self.status,
            },
        }

    def to_failure_record(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "positive": self.positive.to_dict() if self.positive is not None else None,
            "status": self.status,
            "failure_reason": self.failure_reason,
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
    status: str
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
