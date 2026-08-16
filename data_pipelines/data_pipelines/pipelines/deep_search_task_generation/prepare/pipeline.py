import asyncio
import json
import random
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from prefect import flow, get_run_logger, task
from prefect.client.orchestration import get_client
from prefect.concurrency.asyncio import concurrency
from prefect.runtime import flow_run

from data_pipelines.pipelines.deep_search_task_generation.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.models import (
    EntityFactBatchRequest,
    PreparePaths,
)
from data_pipelines.pipelines.deep_search_task_generation.output import (
    entity_to_dict,
    initialize_prepare_output,
    write_json,
    write_jsonl,
    write_retrieval_debug,
)
from data_pipelines.pipelines.deep_search_task_generation.project import phoenix_project
from data_pipelines.pipelines.deep_search_task_generation.prompts import (
    ENTITY_EXTRACTOR_PROMPT,
    FACT_EXTRACTION_PROMPT,
    format_prompt_with_description,
    parse_entities,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker import (
    AsyncRetrieverWorkerClient,
)
from data_pipelines.pipelines.deep_search_task_generation.services import (
    chat_completion,
    load_dataset,
    upload_batch_dataset,
)
from data_pipelines.tracing import (
    configure_tracing,
    object_trace,
    set_span_output,
    stage_span,
)
from ragent_core.retrievers.document import RetrievalResult
from ragent_core.types import Concept
from ragent_core.utils.entity_matching import EntityMatcher

LLM_CONCURRENCY_LIMIT = "deep-search-tasks-openai-llm"
NO_PROGRESS_LIMIT = 5


def default_batch_dataset_name(data_source: str, num_entities: int) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", data_source.lower()).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    return f"deep-search-tasks-{slug or 'source'}-{timestamp}-{num_entities}e"


async def _upsert_llm_concurrency_limit(config: DeepSearchTaskGenerationConfig) -> None:
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            LLM_CONCURRENCY_LIMIT, config.llm_concurrency
        )


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
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("record must be a JSON object")
                entity = Concept(**payload)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Invalid entity record at {path}:{line_number}: {exc}"
                ) from exc

            key = entity.name.strip().lower()
            if not key or key in seen:
                continue
            if entity.data_source != data_source:
                raise ValueError(
                    f"Entity at {path}:{line_number} uses data_source "
                    f"{entity.data_source!r}, expected {data_source!r}."
                )
            if int(entity.doc_id) not in valid_doc_ids:
                raise ValueError(
                    f"Entity at {path}:{line_number} references unknown document "
                    f"ID {entity.doc_id}."
                )
            seen.add(key)
            entities.append(entity)
            if len(entities) >= limit:
                break
    return entities


def _format_documents(
    contents: Iterable[str], titles: Iterable[str], doc_ids: Iterable[int]
) -> str:
    return "".join(
        "<document>\n"
        f"<doc_id>{doc_id}</doc_id>\n"
        f"<title>{title}</title>\n"
        f"<content>{content}</content>\n"
        "</document>\n"
        for content, title, doc_id in zip(contents, titles, doc_ids)
    )


@task(
    name="initialize-deep-search-tasks-prepare-output", retries=0, persist_result=False
)
async def initialize_prepare(
    config: DeepSearchTaskGenerationConfig, run_id: str
) -> PreparePaths:
    return await anyio.to_thread.run_sync(initialize_prepare_output, config, run_id)


@task(name="load-deep-search-tasks-corpus", retries=0, persist_result=False)
async def load_corpus(
    config: DeepSearchTaskGenerationConfig,
) -> tuple[Any, str, str | None]:
    dataset, _, description = await anyio.to_thread.run_sync(
        load_dataset, config.data_source
    )
    # LanceDB table names are keyed by the CLI data-source identifier, not the
    # display/name value returned by the corpus loader.
    return dataset, config.data_source, description


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
                "LLM",
                {"doc_id": doc_id},
                {"llm.model_name": config.entity_model_id},
                project_name=phoenix_project(),
            ) as span:
                async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
                    response = await chat_completion(messages, config.entity_model_id)
                entities = parse_entities(response, data_source_name, doc_id)
                set_span_output(span, [entity_to_dict(item) for item in entities])
            root.set_output({"entity_count": len(entities)})
            return entities, None
        except Exception as exc:
            root.mark_error(exc)
            return [], f"{type(exc).__name__}: {exc}"


def _prepare_requests_sync(
    config: DeepSearchTaskGenerationConfig,
    entity_index: int,
    entity: Concept,
    entity_matcher: EntityMatcher,
    description: str | None,
    retrieval_debug_directory: Path,
    chunks: Sequence[RetrievalResult],
) -> list[EntityFactBatchRequest]:
    ordered_chunks = list(chunks)
    write_retrieval_debug(
        retrieval_debug_directory / f"entity-{entity_index:05d}.json",
        entity.name,
        config.num_chunks_per_entity,
        ordered_chunks,
    )
    ordered_chunks.sort(key=lambda chunk: int(chunk.id))
    requests: list[EntityFactBatchRequest] = []
    size = config.fact_extraction_chunks_per_request
    for group_index, start in enumerate(range(0, len(ordered_chunks), size)):
        group = ordered_chunks[start : start + size]
        doc_ids = tuple(
            int(chunk.metadata.get("document_id", chunk.id)) for chunk in group
        )
        titles = [chunk.title for chunk in group]
        combined_text = "\n".join(chunk.content for chunk in group)
        matched = set(entity_matcher.match(combined_text))
        linked = "\n".join(
            candidate_name
            for candidate_name in entity_matcher.entities
            if candidate_name != entity.name and candidate_name in matched
        )
        prompt = FACT_EXTRACTION_PROMPT.format(
            ENTITY=entity.name,
            ENTITIES=linked,
            PASSAGE=_format_documents(
                [chunk.content for chunk in group], titles, doc_ids
            ),
        )
        requests.append(
            EntityFactBatchRequest(
                key=f"entity-{entity_index:05d}-group-{group_index:05d}",
                entity_name=entity.name,
                data_source=entity.data_source,
                doc_ids=doc_ids,
                chunk_ids=tuple(int(chunk.id) for chunk in group),
                prompt=format_prompt_with_description(prompt, description),
            )
        )
    return requests


@task(
    name="prepare-fact-extraction-requests",
    task_run_name="prepare-facts-{entity_index}",
    retries=0,
    persist_result=False,
)
async def prepare_entity_requests(
    config: DeepSearchTaskGenerationConfig,
    entity_index: int,
    entity: Concept,
    entity_matcher: EntityMatcher,
    description: str | None,
    retrieval_debug_directory: Path,
    chunks: Sequence[RetrievalResult],
) -> list[EntityFactBatchRequest]:
    with object_trace(
        f"fact-request-preparation-{entity_index}-{entity.name}",
        entity_to_dict(entity),
        {"entity.name": entity.name},
        project_name=phoenix_project(),
    ) as root:
        with stage_span(
            root.carrier,
            "prepare_fact_requests",
            "CHAIN",
            entity_to_dict(entity),
            project_name=phoenix_project(),
        ) as span:
            requests = await anyio.to_thread.run_sync(
                _prepare_requests_sync,
                config,
                entity_index,
                entity,
                entity_matcher,
                description,
                retrieval_debug_directory,
                chunks,
            )
            set_span_output(span, {"request_count": len(requests)})
        root.set_output({"request_count": len(requests)})
        return requests


async def retrieve_and_prepare_entity_requests(
    retriever_client: AsyncRetrieverWorkerClient,
    config: DeepSearchTaskGenerationConfig,
    entity_index: int,
    entity: Concept,
    entity_matcher: EntityMatcher,
    description: str | None,
    table_name: str,
    retrieval_debug_directory: Path,
) -> list[EntityFactBatchRequest]:
    chunks = await retriever_client.retrieve(
        entity.name,
        table_name=table_name,
        top_k=config.num_chunks_per_entity,
    )
    return await prepare_entity_requests(
        config,
        entity_index,
        entity,
        entity_matcher,
        description,
        retrieval_debug_directory,
        chunks,
    )


@task(name="upload-fact-extraction-dataset", retries=0, persist_result=False)
async def upload_requests(
    path: Path, dataset_name: str, timeout: float
) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(
        upload_batch_dataset, path, dataset_name, timeout
    )


async def discover_entities_and_prepare_requests(
    config: DeepSearchTaskGenerationConfig,
    retriever_client: AsyncRetrieverWorkerClient,
    paths: PreparePaths,
    failures: list[dict[str, Any]],
    logger: Any,
) -> tuple[str, list[Concept], list[Any]]:
    dataset, table_name, description = await load_corpus(config)
    valid_doc_ids = {int(value) for value in dataset["id"]}
    entities: list[Concept] = []

    if config.entities_file is not None:
        entities = await anyio.to_thread.run_sync(
            load_entities_file,
            config.entities_file,
            table_name,
            valid_doc_ids,
            config.num_entities,
        )
        logger.info(
            "Loaded %d entities from %s; skipping entity extraction.",
            len(entities),
            config.entities_file,
        )
    else:
        rng = random.Random(config.seed)
        seen: set[str] = set()
        no_progress = 0
        max_rounds = max(30, config.num_entities * 2)

        for round_index in range(max_rounds):
            if len(entities) >= config.num_entities or no_progress >= NO_PROGRESS_LIMIT:
                break
            indices = sample_indices(rng, len(dataset), config.sample_size)
            outcomes = await asyncio.gather(
                *(
                    extract_entities_from_document(
                        config,
                        int(dataset[index]["id"]),
                        str(dataset[index].get("title") or ""),
                        str(dataset[index].get("text") or ""),
                        table_name,
                        description,
                    )
                    for index in indices
                )
            )
            added = 0
            for index, (parsed, error) in zip(indices, outcomes):
                if error:
                    failures.append(
                        {
                            "stage": "entity_extraction",
                            "doc_id": int(dataset[index]["id"]),
                            "error": error,
                        }
                    )
                for entity in parsed:
                    key = entity.name.strip().lower()
                    if (
                        not key
                        or key in seen
                        or int(entity.doc_id) not in valid_doc_ids
                        or len(entities) >= config.num_entities
                    ):
                        continue
                    seen.add(key)
                    entities.append(entity)
                    added += 1
            no_progress = no_progress + 1 if added == 0 else 0
            logger.info(
                "Entity round %d retained %d new entities (%d/%d).",
                round_index + 1,
                added,
                len(entities),
                config.num_entities,
            )

    write_jsonl(paths.entities, (entity_to_dict(entity) for entity in entities))
    entity_matcher = EntityMatcher(entity.name for entity in entities)
    request_outcomes = await asyncio.gather(
        *(
            retrieve_and_prepare_entity_requests(
                retriever_client,
                config,
                index,
                entity,
                entity_matcher,
                description,
                table_name,
                paths.retrieval_debug_directory,
            )
            for index, entity in enumerate(entities)
        ),
        return_exceptions=True,
    )
    return table_name, entities, request_outcomes


@flow(
    name="deep-search-tasks-prepare",
    flow_run_name="deep-search-tasks-prepare-{config.data_source}",
    retries=0,
    persist_result=False,
)
async def prepare_deep_search_tasks_flow(
    config: DeepSearchTaskGenerationConfig,
) -> dict[str, Any]:
    logger = get_run_logger()
    run_id = str(flow_run.id)
    tracing = configure_tracing(project_name=phoenix_project())
    await _upsert_llm_concurrency_limit(config)
    paths = await initialize_prepare(config, run_id)
    failures: list[dict[str, Any]] = []
    async with AsyncRetrieverWorkerClient(
        client_id=run_id,
        port=config.retriever_worker_port,
    ) as retriever_client:
        (
            table_name,
            entities,
            request_outcomes,
        ) = await discover_entities_and_prepare_requests(
            config,
            retriever_client,
            paths,
            failures,
            logger,
        )
        retriever_worker_info = await retriever_client.health()
    requests: list[EntityFactBatchRequest] = []
    crashed = 0
    for entity, outcome in zip(entities, request_outcomes):
        if isinstance(outcome, BaseException):
            crashed += 1
            failures.append(
                {
                    "stage": "fact_request_preparation",
                    "entity": entity.name,
                    "error": f"{type(outcome).__name__}: {outcome}",
                    "crashed": True,
                }
            )
        elif not outcome:
            failures.append(
                {
                    "stage": "fact_request_preparation",
                    "entity": entity.name,
                    "error": "No chunks were retrieved.",
                }
            )
        else:
            requests.extend(outcome)
    write_jsonl(
        paths.fact_requests,
        (request.to_fireworks_record() for request in requests),
    )
    write_jsonl(paths.failures, failures)

    resolved_dataset_name = default_batch_dataset_name(
        config.data_source, len(entities)
    )
    upload_payload: dict[str, Any] = {}
    status = "empty" if not requests else "prepared"
    upload_error: BaseException | None = None
    if requests:
        try:
            upload_payload = await upload_requests(
                paths.fact_requests, resolved_dataset_name, config.upload_timeout
            )
            status = "uploaded"
        except BaseException as exc:
            upload_error = exc
            status = "upload_failed"
            failures.append(
                {
                    "stage": "dataset_upload",
                    "error": f"{type(exc).__name__}: {exc}",
                    "crashed": True,
                }
            )
            write_jsonl(paths.failures, failures)

    metadata = {
        "status": status,
        "config": config.model_dump(mode="json"),
        "prefect_flow_run_id": run_id,
        "phoenix_project": tracing.project_name,
        "retriever_worker": retriever_worker_info,
        "data_source": config.data_source,
        "table_name": table_name,
        "requested_entities": config.num_entities,
        "retained_entities": len(entities),
        "fact_request_count": len(requests),
        "failure_count": len(failures),
        "crashed_entities": crashed,
        "fireworks": {
            "input_dataset_name": resolved_dataset_name if requests else None,
            "upload_payload": upload_payload,
        },
        "paths": {
            "prepare_run_directory": str(paths.directory),
            "entities": str(paths.entities),
            "retrieval_debug_directory": str(paths.retrieval_debug_directory),
            "fact_requests": str(paths.fact_requests),
            "failures": str(paths.failures),
        },
    }
    write_json(paths.metadata, metadata)
    tracing.force_flush()
    if upload_error is not None:
        raise RuntimeError(
            f"Fireworks dataset upload failed; metadata written to {paths.metadata}."
        ) from upload_error
    if crashed:
        raise RuntimeError(
            f"{crashed} fact-request preparation task(s) crashed; metadata written "
            f"to {paths.metadata}."
        )
    return metadata
