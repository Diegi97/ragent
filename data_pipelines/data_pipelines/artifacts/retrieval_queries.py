from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)

from ragent_core.retrievers.document import (
    Document,
    DocumentId,
    RetrievalResult,
)

StrictDocumentId = Annotated[DocumentId, BeforeValidator(Document.validate_id)]


@dataclass(frozen=True)
class RetrievalChunk:
    id: StrictDocumentId
    title: str = ""
    text: str = ""
    document_id: StrictDocumentId | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    sources: tuple[str, ...] = field(default_factory=tuple)
    source_ranks: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Document) -> "RetrievalChunk":
        return cls(
            id=document.id,
            title=document.title,
            text=document.content,
            document_id=document.document_id,
            metadata=document.metadata,
        )

    @classmethod
    def from_result(cls, result: RetrievalResult, rank: int) -> "RetrievalChunk":
        return cls(
            id=result.id,
            title=result.title,
            text=result.content,
            document_id=result.parent_document_id,
            metadata=result.metadata,
            score=result.score,
            sources=("hybrid",),
            source_ranks={"hybrid": rank},
        )

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


class TrainingRecord(BaseModel):
    query: str = Field(min_length=1)
    positive: RetrievalChunk
    hard_negatives: tuple[RetrievalChunk, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def require_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def require_parent_document(self) -> "TrainingRecord":
        if self.positive.document_id is None:
            raise ValueError("positive.document_id is required")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "positive": self.positive.to_dict(),
            "hard_negatives": [chunk.to_dict() for chunk in self.hard_negatives],
            "metadata": self.metadata,
        }
