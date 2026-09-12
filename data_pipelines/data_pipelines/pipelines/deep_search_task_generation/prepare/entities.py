import random
from pathlib import Path

from prefect import task
from prefect.concurrency.asyncio import concurrency
from pydantic import TypeAdapter

from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import Concept
from data_pipelines.pipelines.deep_search_task_generation.project import (
    LLM_CONCURRENCY_LIMIT,
    phoenix_project,
)
from data_pipelines.pipelines.deep_search_task_generation.prompts import (
    format_prompt_with_description,
)
from data_pipelines.pipelines.deep_search_task_generation.prompts.entities import (
    ENTITY_EXTRACTOR_PROMPT,
    parse_entities,
)
from data_pipelines.providers.openai import chat_completion
from data_pipelines.tracing import (
    SpanKind,
    object_trace,
    set_span_output,
    stage_span,
)

NO_PROGRESS_LIMIT = 5
ENTITY_RECORD_ADAPTER = TypeAdapter(Concept)


def sample_indices(
    rng: random.Random,
    population: int,
    sample_size: int,
) -> list[int]:
    if population <= 0:
        return []
    return rng.sample(range(population), k=min(sample_size, population))


def load_entities_file(
    path: Path,
    data_source: str,
    valid_doc_ids: set[int],
    limit: int,
) -> list[Concept]:
    entities: list[Concept] = []
    seen: set[str] = set()
    if limit == 0:
        return entities

    with path.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                entity = ENTITY_RECORD_ADAPTER.validate_json(
                    line, strict=True, extra="forbid"
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid entity record at {path}:{line_number}: {exc}"
                ) from exc

            if entity.data_source != data_source:
                raise ValueError(
                    f"Entity at {path}:{line_number} uses data_source "
                    f"{entity.data_source!r}, expected {data_source!r}."
                )
            if entity.doc_id not in valid_doc_ids:
                raise ValueError(
                    f"Entity at {path}:{line_number} references unknown document "
                    f"ID {entity.doc_id}."
                )
            normalized_entity_name = entity.normalized_name
            if not normalized_entity_name or normalized_entity_name in seen:
                continue
            seen.add(normalized_entity_name)
            entities.append(entity)
            if len(entities) >= limit:
                break
    return entities


@task(
    name="extract-entities-from-document",
    task_run_name="extract-entities-{doc_id}",
    retries=0,
    persist_result=False,
)
async def extract_entities_from_document(
    config: DeepSearchTaskGenerationConfig,
    doc_id: int,
    title: str,
    content: str,
    data_source_name: str,
    description: str | None,
) -> tuple[list[Concept], str | None]:
    trace_name = f"entity-extraction-{doc_id}"
    with object_trace(
        trace_name,
        {"doc_id": doc_id, "title": title},
        {"document.id": str(doc_id), "llm.model_name": config.entity_model_id},
        project_name=phoenix_project(),
    ) as root:
        prompt = ENTITY_EXTRACTOR_PROMPT.format(
            DOC_ID=doc_id, TITLE=title, CONTENT=content
        )
        messages = [
            {
                "role": "user",
                "content": format_prompt_with_description(prompt, description),
            }
        ]
        try:
            with stage_span(
                root.carrier,
                "extract_entities",
                SpanKind.LLM,
                {"doc_id": doc_id},
                {"llm.model_name": config.entity_model_id},
                project_name=phoenix_project(),
            ) as span:
                async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
                    response = await chat_completion(messages, config.entity_model_id)
                entities = parse_entities(response.content, data_source_name, doc_id)
                set_span_output(span, [item.to_dict() for item in entities])
            root.set_output({"entity_count": len(entities)})
            return entities, None
        except Exception as exc:
            root.mark_error(exc)
            return [], f"{type(exc).__name__}: {exc}"
