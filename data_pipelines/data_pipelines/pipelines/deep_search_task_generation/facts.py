from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ragent_core.artifacts.question_rubric import SupportingDocumentId

ENTITY_FACTS_FILENAME = "entity_facts.jsonl"


@dataclass(frozen=True)
class ExtractedFact:
    statement: str
    doc_ids: list[int]
    fact_id: int = 0
    mentioned_entities: list[str] = field(default_factory=list)


class FactRequestMetadata(BaseModel):
    custom_id: str = Field(min_length=1)
    entity_name: str = Field(min_length=1)
    data_source: str = Field(min_length=1)
    doc_ids: tuple[SupportingDocumentId, ...] = Field(min_length=1)
    chunk_ids: tuple[SupportingDocumentId, ...] = Field(min_length=1)

    @field_validator("custom_id", "entity_name", "data_source")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Fact request identity fields must not be blank")
        return value.strip()

    @classmethod
    def from_request(cls, request: "EntityFactBatchRequest") -> "FactRequestMetadata":
        return cls(
            custom_id=request.key,
            entity_name=request.entity_name,
            data_source=request.data_source,
            doc_ids=request.doc_ids,
            chunk_ids=request.chunk_ids,
        )


@dataclass(frozen=True)
class EntityFactBatchRequest:
    key: str
    entity_name: str
    data_source: str
    doc_ids: tuple[int, ...]
    chunk_ids: tuple[int, ...]
    prompt: str

    def to_fireworks_record(
        self,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": [{"role": "user", "content": self.prompt}]}
        return {
            **FactRequestMetadata.from_request(self).model_dump(mode="json"),
            "body": body,
        }


@dataclass(frozen=True)
class EntityFactMemoryRecord:
    entity_name: str
    data_source: str
    entity_doc_ids: tuple[int, ...]
    facts: tuple[ExtractedFact, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "data_source": self.data_source,
            "entity_doc_ids": list(self.entity_doc_ids),
            "facts": [
                {
                    "statement": fact.statement,
                    "doc_ids": fact.doc_ids,
                    "fact_id": fact.fact_id,
                    "mentioned_entities": fact.mentioned_entities,
                }
                for fact in self.facts
            ],
        }
