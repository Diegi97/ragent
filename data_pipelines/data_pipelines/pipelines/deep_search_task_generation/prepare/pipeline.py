import asyncio
import random
from pathlib import Path
from typing import Any

import anyio
from prefect import flow, get_run_logger, task
from prefect.client.orchestration import get_client
from prefect.runtime import flow_run
from pydantic import TypeAdapter

from data_pipelines.artifacts.io import write_json, write_jsonl
from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactBatchRequest,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.entities import (
    NO_PROGRESS_LIMIT,
    extract_entities_from_document,
    load_entities_file,
    sample_indices,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.entity_matching import (
    EntityMatcher,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import (
    Concept,
    PrepareFailure,
    PreparePaths,
    PrepareRunMetadata,
    PrepareStage,
    PrepareStatus,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.output import (
    initialize_prepare_output,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.requests import (
    FactRequestBuilder,
    retrieve_and_prepare_entity_requests,
)
from data_pipelines.pipelines.deep_search_task_generation.project import (
    LLM_CONCURRENCY_LIMIT,
    phoenix_project,
)
from data_pipelines.pipelines.deep_search_task_generation.retrieval_worker.client import (
    AsyncRetrieverWorkerClient,
)
from data_pipelines.providers.fireworks import (
    FireworksUploadResult,
    fact_dataset_name,
    upload_batch_dataset,
)
from data_pipelines.tracing import (
    configure_tracing,
)
from ragent_core.artifacts.question_rubric import SupportingDocumentId
from ragent_core.data_sources import load_corpus as load_source_corpus


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
    failures: list[PrepareFailure] = []
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
        retriever_worker_info = (await retriever_client.health()).model_dump(
            mode="json"
        )
    requests: list[EntityFactBatchRequest] = []
    crashed = 0
    for entity, outcome in zip(entities, request_outcomes):
        if isinstance(outcome, BaseException):
            crashed += 1
            failures.append(
                PrepareFailure(
                    stage=PrepareStage.FACT_REQUEST_PREPARATION,
                    entity=entity.name,
                    error=f"{type(outcome).__name__}: {outcome}",
                    crashed=True,
                )
            )
        elif not outcome:
            failures.append(
                PrepareFailure(
                    stage=PrepareStage.FACT_REQUEST_PREPARATION,
                    entity=entity.name,
                    error="No chunks were retrieved.",
                )
            )
        else:
            requests.extend(outcome)
    await asyncio.to_thread(
        write_jsonl,
        paths.fact_requests,
        (request.to_fireworks_record() for request in requests),
    )
    await asyncio.to_thread(write_jsonl, paths.failures, failures)

    resolved_dataset_name = fact_dataset_name(config.data_source, run_id)
    upload_result: FireworksUploadResult | None = None
    status = PrepareStatus.EMPTY if not requests else PrepareStatus.PREPARED
    upload_error: BaseException | None = None
    if requests:
        try:
            upload_result = await upload_requests(
                paths.fact_requests, resolved_dataset_name, config.upload_timeout
            )
            status = PrepareStatus.UPLOADED
        except BaseException as exc:
            upload_error = exc
            status = PrepareStatus.UPLOAD_FAILED
            failures.append(
                PrepareFailure(
                    stage=PrepareStage.DATASET_UPLOAD,
                    error=f"{type(exc).__name__}: {exc}",
                    crashed=True,
                )
            )
            await asyncio.to_thread(write_jsonl, paths.failures, failures)

    extraction_shortfall = len(entities) < config.num_entities and any(
        failure["stage"] == PrepareStage.ENTITY_EXTRACTION for failure in failures
    )
    if crashed or extraction_shortfall:
        status = PrepareStatus.FAILED
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
            "upload_payload": upload_result.to_dict() if upload_result else {},
        },
        "paths": {
            "prepare_run_directory": str(paths.directory),
            "fact_responses": str(paths.fact_responses),
            "entities": str(paths.entities),
            "retrieval_debug_directory": str(paths.retrieval_debug_directory),
            "fact_requests": str(paths.fact_requests),
            "failures": str(paths.failures),
        },
    }
    metadata = PrepareRunMetadata.model_validate(metadata).model_dump(
        mode="json", exclude_unset=True
    )
    await asyncio.to_thread(write_json, paths.metadata, metadata)
    await asyncio.to_thread(tracing.force_flush)
    if upload_error is not None:
        raise RuntimeError(
            f"Fireworks dataset upload failed; metadata written to {paths.metadata}."
        ) from upload_error
    if extraction_shortfall:
        raise RuntimeError(
            f"Entity extraction failed to reach its target after provider errors; "
            f"metadata written to {paths.metadata}."
        )
    if crashed:
        raise RuntimeError(
            f"{crashed} fact-request preparation task(s) crashed; metadata written "
            f"to {paths.metadata}."
        )
    return metadata


async def discover_entities_and_prepare_requests(
    config: DeepSearchTaskGenerationConfig,
    retriever_client: AsyncRetrieverWorkerClient,
    paths: PreparePaths,
    failures: list[PrepareFailure],
    logger: Any,
) -> tuple[str, list[Concept], list[Any]]:
    dataset, table_name, description = await load_corpus(config)
    valid_doc_ids = set(dataset["id"])
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
                        dataset[index]["id"],
                        dataset[index]["title"],
                        dataset[index]["text"],
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
                        PrepareFailure(
                            stage=PrepareStage.ENTITY_EXTRACTION,
                            doc_id=dataset[index]["id"],
                            error=error,
                        )
                    )
                for entity in parsed:
                    normalized_entity_name = entity.normalized_name
                    if (
                        not normalized_entity_name
                        or normalized_entity_name in seen
                        or entity.doc_id not in valid_doc_ids
                        or len(entities) >= config.num_entities
                    ):
                        continue
                    seen.add(normalized_entity_name)
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

    await asyncio.to_thread(
        write_jsonl, paths.entities, (entity.to_dict() for entity in entities)
    )
    request_builder = FactRequestBuilder(
        EntityMatcher(entity.name for entity in entities)
    )
    request_outcomes = await asyncio.gather(
        *(
            retrieve_and_prepare_entity_requests(
                retriever_client,
                config,
                index,
                entity,
                request_builder,
                description,
                table_name,
                paths.retrieval_debug_directory,
            )
            for index, entity in enumerate(entities)
        ),
        return_exceptions=True,
    )
    return table_name, entities, request_outcomes


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
        load_source_corpus, config.data_source
    )
    TypeAdapter(list[SupportingDocumentId]).validate_python(list(dataset["id"]))
    # Catalog table names are keyed by the CLI data-source identifier, not the
    # display/name value returned by the corpus loader.
    return dataset, config.data_source, description


@task(name="upload-fact-extraction-dataset", retries=0, persist_result=False)
async def upload_requests(
    path: Path, dataset_name: str, timeout: float
) -> FireworksUploadResult:
    return await anyio.to_thread.run_sync(
        upload_batch_dataset, path, dataset_name, timeout
    )


async def _upsert_llm_concurrency_limit(config: DeepSearchTaskGenerationConfig) -> None:
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            LLM_CONCURRENCY_LIMIT, config.llm_concurrency
        )
