from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_pipelines.pipelines.deep_search_task_generation.prompts import ExtractedFact


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
            "custom_id": self.key,
            "body": body,
            "entity_name": self.entity_name,
            "data_source": self.data_source,
            "doc_ids": list(self.doc_ids),
            "chunk_ids": list(self.chunk_ids),
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


@dataclass
class ParseDiagnostics:
    malformed_lines: int = 0
    missing_custom_ids: int = 0
    missing_choices: int = 0
    unmatched_responses: int = 0
    duplicate_custom_ids: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "malformed_lines": self.malformed_lines,
            "missing_custom_ids": self.missing_custom_ids,
            "missing_choices": self.missing_choices,
            "unmatched_responses": self.unmatched_responses,
            "duplicate_custom_ids": self.duplicate_custom_ids,
        }
