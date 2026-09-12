from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
    ExtractedFact,
    FactRequestMetadata,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output.diagnostics import (
    ParseDiagnostics,
)
from data_pipelines.pipelines.deep_search_task_generation.prompts.facts import (
    parse_extracted_facts,
)


@dataclass
class EntityFactAccumulator:
    entity_name: str
    data_source: str
    doc_ids: list[int] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    statements: set[str] = field(default_factory=set)

    def add(
        self,
        facts: Sequence[ExtractedFact],
        metadata: FactRequestMetadata,
        diagnostics: ParseDiagnostics,
    ) -> None:
        allowed = set(metadata.doc_ids)
        for fact in facts:
            if not fact.doc_ids or not set(fact.doc_ids) <= allowed:
                diagnostics.unsupported_facts += 1
                diagnostics.failures.append(
                    {
                        "stage": "batch_output_join",
                        "custom_id": metadata.custom_id,
                        "error": "Fact references documents outside its request",
                        "statement": fact.statement,
                        "doc_ids": fact.doc_ids,
                    }
                )
                continue
            statement_key = " ".join(fact.statement.lower().split())
            if statement_key not in self.statements:
                self.statements.add(statement_key)
                self.facts.append(fact)
        for document_id in metadata.doc_ids:
            if document_id not in self.doc_ids:
                self.doc_ids.append(document_id)

    def to_record(self) -> EntityFactMemoryRecord:
        return EntityFactMemoryRecord(
            self.entity_name, self.data_source, tuple(self.doc_ids), tuple(self.facts)
        )


def build_entity_facts_from_batch_output(
    responses: Mapping[str, str],
    metadata_by_key: Mapping[str, FactRequestMetadata],
    diagnostics: ParseDiagnostics | None = None,
) -> list[EntityFactMemoryRecord]:
    diagnostics = diagnostics or ParseDiagnostics()
    records: dict[tuple[str, str], EntityFactAccumulator] = {}
    for custom_id, content in responses.items():
        metadata = metadata_by_key.get(custom_id)
        if metadata is None:
            diagnostics.unmatched_responses += 1
            diagnostics.failures.append(
                {
                    "stage": "batch_output_join",
                    "custom_id": custom_id,
                    "error": "No matching fact request metadata",
                }
            )
            continue
        try:
            facts = parse_extracted_facts(content, entity_name=metadata.entity_name)
        except ValueError as exc:
            diagnostics.invalid_contents += 1
            diagnostics.failures.append(
                {
                    "stage": "batch_output_join",
                    "custom_id": custom_id,
                    "error": str(exc),
                }
            )
            continue
        key = (metadata.data_source, metadata.entity_name)
        accumulator = records.setdefault(
            key, EntityFactAccumulator(metadata.entity_name, metadata.data_source)
        )
        accumulator.add(facts, metadata, diagnostics)
    for custom_id in set(metadata_by_key).difference(responses):
        diagnostics.missing_responses += 1
        diagnostics.failures.append(
            {
                "stage": "batch_output_join",
                "custom_id": custom_id,
                "error": "No valid response was found for this fact request",
            }
        )
    return [accumulator.to_record() for accumulator in records.values()]
