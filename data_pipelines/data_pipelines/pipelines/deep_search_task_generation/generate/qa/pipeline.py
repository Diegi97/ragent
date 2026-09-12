import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anyio
from prefect import flow, task
from prefect.client.orchestration import get_client
from prefect.runtime import flow_run

from data_pipelines.artifacts.append import append_jsonl
from data_pipelines.artifacts.inputs import count_jsonl, read_json
from data_pipelines.artifacts.io import write_json, write_jsonl
from data_pipelines.pipelines.deep_search_task_generation.generate import (
    GenerationStatus,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.fact_extraction_output import (
    parse_fact_output,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.config import (
    QAGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.entity import (
    generate_deep_search_qas_for_entity_flow,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    EntityQASummary,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.output import (
    initialize_finalize_output,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.config import (
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import (
    PreparePaths,
    PrepareRunMetadata,
)
from data_pipelines.pipelines.deep_search_task_generation.project import (
    LLM_CONCURRENCY_LIMIT,
    phoenix_project,
)
from data_pipelines.providers.fireworks import download_output
from data_pipelines.tracing import (
    configure_tracing,
)
from ragent_core.data_sources import load_corpus as load_source_corpus


@flow(
    name="deep-search-tasks-generate-qas",
    flow_run_name="deep-search-tasks-generate-qas-{batch_output_dataset_name}",
    retries=0,
    persist_result=False,
)
async def generate_deep_search_qas_flow(
    prepare_run_directory: Path,
    batch_output_dataset_name: str,
    qa_pairs_per_entity: int = QAGenerationConfig.model_fields[
        "qa_pairs_per_entity"
    ].default,
    qa_model_id: str = QAGenerationConfig.model_fields["qa_model_id"].default,
    complex_pair_ratio: float = QAGenerationConfig.model_fields[
        "complex_pair_ratio"
    ].default,
    max_qa_generation_attempts: int = QAGenerationConfig.model_fields[
        "max_qa_generation_attempts"
    ].default,
    llm_concurrency: int = QAGenerationConfig.model_fields["llm_concurrency"].default,
    download_timeout: float = QAGenerationConfig.model_fields[
        "download_timeout"
    ].default,
) -> dict[str, Any]:
    generation_config = QAGenerationConfig(
        qa_pairs_per_entity=qa_pairs_per_entity,
        qa_model_id=qa_model_id,
        complex_pair_ratio=complex_pair_ratio,
        max_qa_generation_attempts=max_qa_generation_attempts,
        llm_concurrency=llm_concurrency,
        download_timeout=download_timeout,
    )
    qa_model_id = generation_config.qa_model_id
    prepare_run_directory = prepare_run_directory.expanduser().resolve()
    prepare_paths = PreparePaths.in_directory(prepare_run_directory)
    prepare_metadata_path = prepare_paths.metadata
    prepare_metadata = PrepareRunMetadata.model_validate(
        await asyncio.to_thread(read_json, prepare_metadata_path)
    )
    config = prepare_metadata.config
    run_id = str(flow_run.id)
    tracing = configure_tracing(project_name=phoenix_project())
    await _upsert_llm_concurrency_limit(llm_concurrency)
    paths = await asyncio.to_thread(
        initialize_finalize_output, prepare_run_directory, run_id
    )
    finalize_config = generation_config.model_dump(mode="json")

    stage = "download_batch_output"
    try:
        downloaded = await download_output(
            batch_output_dataset_name, paths.raw_directory, download_timeout
        )
        stage = "parse_batch_output"
        responses, entity_facts, diagnostics = await parse_fact_output(
            downloaded, prepare_paths.fact_requests
        )
        await asyncio.to_thread(
            write_jsonl,
            paths.entity_facts,
            (record.to_dict() for record in entity_facts),
        )
        await asyncio.to_thread(write_jsonl, paths.failures, diagnostics.failures)
        stage = "load_corpus"
        _, _, description = await load_corpus(config)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        await asyncio.to_thread(
            append_jsonl,
            paths.failures,
            {"stage": stage, "error": error, "crashed": True},
            paths.lock,
        )
        failed_metadata = {
            "status": GenerationStatus.FAILED,
            "config": config.model_dump(mode="json"),
            "finalize_config": finalize_config,
            "prefect_flow_run_id": run_id,
            "phoenix_project": tracing.project_name,
            "prepare_run_directory": str(prepare_run_directory),
            "prepare_prefect_flow_run_id": prepare_metadata.prefect_flow_run_id,
            "fireworks": {
                "input_dataset_name": prepare_metadata.fireworks.input_dataset_name,
                "output_dataset_name": batch_output_dataset_name,
            },
            "failed_stage": stage,
            "error": error,
            "paths": {
                "finalize_run_directory": str(paths.directory),
                "raw_directory": str(paths.raw_directory),
                "entity_facts": str(paths.entity_facts),
                "qas": str(paths.qas),
                "failures": str(paths.failures),
            },
        }
        await asyncio.to_thread(write_json, paths.metadata, failed_metadata)
        await asyncio.to_thread(tracing.force_flush)
        raise RuntimeError(
            f"Deep-search QA generation failed during {stage}; metadata written to "
            f"{paths.metadata}."
        ) from exc
    outcomes = await asyncio.gather(
        *(
            generate_deep_search_qas_for_entity_flow(
                index,
                record,
                description,
                paths,
                qa_pairs_per_entity,
                qa_model_id,
                complex_pair_ratio,
                max_qa_generation_attempts,
            )
            for index, record in enumerate(entity_facts)
        ),
        return_exceptions=True,
    )
    summaries: list[EntityQASummary] = []
    crashes: list[dict[str, Any]] = []
    for record, outcome in zip(entity_facts, outcomes):
        if isinstance(outcome, BaseException):
            error = f"{type(outcome).__name__}: {outcome}"
            crashes.append({"entity": record.entity_name, "error": error})
            await asyncio.to_thread(
                append_jsonl,
                paths.failures,
                {
                    "stage": "qa_generation",
                    "entity": record.entity_name,
                    "error": error,
                    "crashed": True,
                },
                paths.lock,
            )
            summaries.append(
                EntityQASummary(
                    entity_name=record.entity_name,
                    requested=qa_pairs_per_entity,
                    generated=0,
                    crashed=True,
                    error=error,
                )
            )
        else:
            summaries.append(outcome)
    qa_count = count_jsonl(paths.qas)
    failure_count = count_jsonl(paths.failures)
    if crashes:
        status = GenerationStatus.FAILED
    elif not entity_facts:
        status = (
            GenerationStatus.FAILED
            if diagnostics.has_integrity_errors
            else GenerationStatus.EMPTY
        )
    else:
        status = GenerationStatus.for_output(
            qa_count,
            len(entity_facts) * qa_pairs_per_entity,
            diagnostics.has_integrity_errors,
        )
    metadata = {
        "status": status,
        "config": config.model_dump(mode="json"),
        "finalize_config": finalize_config,
        "prefect_flow_run_id": run_id,
        "phoenix_project": tracing.project_name,
        "prepare_run_directory": str(prepare_run_directory),
        "prepare_prefect_flow_run_id": prepare_metadata.prefect_flow_run_id,
        "fireworks": {
            "input_dataset_name": prepare_metadata.fireworks.input_dataset_name,
            "output_dataset_name": batch_output_dataset_name,
        },
        "downloaded_file_count": len(downloaded),
        "batch_response_count": len(responses),
        "entity_fact_count": len(entity_facts),
        "qa_count": qa_count,
        "requested_qa_count": len(entity_facts) * qa_pairs_per_entity,
        "failure_count": failure_count,
        "crashed_entities": crashes,
        "parse_diagnostics": diagnostics.to_dict(),
        "entity_summaries": [asdict(summary) for summary in summaries],
        "paths": {
            "finalize_run_directory": str(paths.directory),
            "raw_directory": str(paths.raw_directory),
            "entity_facts": str(paths.entity_facts),
            "qas": str(paths.qas),
            "failures": str(paths.failures),
        },
    }
    await asyncio.to_thread(write_json, paths.metadata, metadata)
    await asyncio.to_thread(
        configure_tracing(project_name=phoenix_project()).force_flush
    )
    if crashes:
        raise RuntimeError(
            f"{len(crashes)} entity QA flow(s) crashed; metadata written to "
            f"{paths.metadata}."
        )
    return metadata


@task(name="load-deep-search-tasks-qa-corpus", retries=0, persist_result=False)
async def load_corpus(
    config: DeepSearchTaskGenerationConfig,
) -> tuple[Any, str, str | None]:
    dataset, name, description = await anyio.to_thread.run_sync(
        load_source_corpus, config.data_source
    )
    return dataset, name or config.data_source, description


async def _upsert_llm_concurrency_limit(limit: int) -> None:
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            LLM_CONCURRENCY_LIMIT, limit
        )
