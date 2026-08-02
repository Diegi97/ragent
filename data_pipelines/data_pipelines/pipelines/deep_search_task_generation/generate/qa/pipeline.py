import asyncio
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anyio
from prefect import flow, task
from prefect.client.orchestration import get_client
from prefect.concurrency.asyncio import concurrency
from prefect.runtime import flow_run

from data_pipelines.pipelines.deep_search_task_generation.config import (
    DEFAULT_QA_MODEL_ID,
    DeepSearchTaskGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.models import (
    EntityQASummary,
    FinalizePaths,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.output import (
    initialize_finalize_output,
    qa_to_dict,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.qa.prompts import (
    FACT_TO_QA_PROMPT,
    parse_fact_grounded_qas,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.shared import (
    download_output,
    parse_fact_output,
)
from data_pipelines.pipelines.deep_search_task_generation.models import (
    EntityFactMemoryRecord,
)
from data_pipelines.pipelines.deep_search_task_generation.output import (
    append_jsonl,
    count_jsonl,
    read_json,
    write_json,
    write_jsonl,
)
from data_pipelines.pipelines.deep_search_task_generation.project import phoenix_project
from data_pipelines.pipelines.deep_search_task_generation.prompts import (
    ExtractedFact,
    format_prompt_with_description,
)
from data_pipelines.pipelines.deep_search_task_generation.services import (
    chat_completion,
    load_dataset,
)
from data_pipelines.tracing import (
    configure_tracing,
    get_tracing,
    object_trace,
    set_span_error,
    set_span_output,
    stage_span,
)
from ragent_core.types import QA

LLM_CONCURRENCY_LIMIT = "deep-search-tasks-openai-llm"


async def _upsert_llm_concurrency_limit(limit: int) -> None:
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            LLM_CONCURRENCY_LIMIT, limit
        )


def normalize_candidate_doc_ids(
    doc_ids: Sequence[int], allowed_doc_ids: set[int]
) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for value in doc_ids:
        candidate = int(value)
        if candidate in allowed_doc_ids and candidate not in seen:
            normalized.append(candidate)
            seen.add(candidate)
    return normalized


def _format_facts(facts: Sequence[ExtractedFact]) -> str:
    blocks: list[str] = []
    for fact in facts:
        lines = ["<fact>"]
        if fact.fact_id > 0:
            lines.append(f"<fact_id>{fact.fact_id}</fact_id>")
        lines.extend(
            [
                f"<statement>{fact.statement}</statement>",
                f"<doc_ids>{','.join(str(value) for value in fact.doc_ids)}</doc_ids>",
            ]
        )
        if fact.mentioned_entities:
            lines.append(
                "<mentioned_entities>"
                + ", ".join(fact.mentioned_entities)
                + "</mentioned_entities>"
            )
        lines.append("</fact>")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


@task(name="load-deep-search-tasks-qa-corpus", retries=0, persist_result=False)
async def load_corpus(
    config: DeepSearchTaskGenerationConfig,
) -> tuple[Any, str, str | None]:
    dataset, name, description = await anyio.to_thread.run_sync(
        load_dataset, config.data_source
    )
    return dataset, name or config.data_source, description


@task(name="generate-entity-qa-candidate", retries=0, persist_result=False)
async def generate_qa_candidate(
    qa_model_id: str,
    entity_name: str,
    facts: Sequence[ExtractedFact],
    description: str | None,
    must_be_complex: bool,
    trace_carrier: dict[str, str],
) -> tuple[QA | None, str | None]:
    prompt = FACT_TO_QA_PROMPT.format(
        ENTITY=entity_name,
        FACTS=_format_facts(facts),
        COMPLEXITY_TARGET="complex" if must_be_complex else "simple",
    )
    with stage_span(
        trace_carrier,
        "generate_qa_candidate",
        "LLM",
        {"entity": entity_name, "complex": must_be_complex},
        {"llm.model_name": qa_model_id},
        project_name=phoenix_project(),
    ) as span:
        try:
            async with concurrency(LLM_CONCURRENCY_LIMIT, strict=True):
                response = await chat_completion(
                    [
                        {
                            "role": "user",
                            "content": format_prompt_with_description(
                                prompt, description
                            ),
                        }
                    ],
                    qa_model_id,
                )
            parsed = parse_fact_grounded_qas(response)
            if not parsed:
                reason = "Model response contained no valid QA pair."
                set_span_error(span, reason)
                return None, reason
            candidate = parsed[0]
            allowed = {doc_id for fact in facts for doc_id in fact.doc_ids}
            doc_ids = normalize_candidate_doc_ids(candidate.doc_ids, allowed)
            if len(doc_ids) < 2:
                reason = "QA pair is not grounded in at least two allowed documents."
                set_span_error(span, reason)
                return None, reason
            qa = QA(
                question=candidate.question.strip(),
                answer=candidate.answer.strip(),
                doc_ids=doc_ids,
                info={
                    "entity": entity_name,
                    "complexity": "complex" if must_be_complex else "simple",
                    "num_docs": len(doc_ids),
                    "num_facts_for_entity": len(facts),
                    "facts": [asdict(fact) for fact in facts],
                },
            )
            if not qa.question or not qa.answer:
                reason = "QA pair contains an empty question or answer."
                set_span_error(span, reason)
                return None, reason
            set_span_output(span, qa_to_dict(qa))
            return qa, None
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            set_span_error(span, reason)
            return None, reason


@task(name="append-entity-fact-record", retries=0, persist_result=False)
async def append_output_record(
    path: Path, value: dict[str, Any], lock_path: Path
) -> None:
    await anyio.to_thread.run_sync(append_jsonl, path, value, lock_path)


@flow(
    name="deep-search-tasks-generate-qas-for-entity",
    flow_run_name="entity-qa-{entity_index}-{entity_fact.entity_name}",
    retries=0,
    persist_result=False,
)
async def generate_deep_search_qas_for_entity_flow(
    entity_index: int,
    entity_fact: EntityFactMemoryRecord,
    description: str | None,
    paths: FinalizePaths,
    qa_pairs_per_entity: int,
    qa_model_id: str,
    complex_pair_ratio: float,
    max_qa_generation_attempts: int,
) -> EntityQASummary:
    target = qa_pairs_per_entity
    required_complex = min(target, math.ceil(target * complex_pair_ratio))
    accepted: list[QA] = []
    seen_questions: set[str] = set()
    errors: list[str] = []
    with object_trace(
        f"deep-search-tasks-qa-{entity_index}-{entity_fact.entity_name}",
        entity_fact.to_dict(),
        {"entity.name": entity_fact.entity_name, "requested_qas": target},
        project_name=phoenix_project(),
    ) as root:
        for _ in range(max_qa_generation_attempts):
            if len(accepted) >= target:
                break
            current_complex = sum(
                qa.info.get("complexity") == "complex" for qa in accepted
            )
            remaining_slots = target - len(accepted)
            remaining_complex = max(0, required_complex - current_complex)
            targets = [True] * min(remaining_complex, remaining_slots)
            targets.extend([False] * (remaining_slots - len(targets)))
            outcomes = await asyncio.gather(
                *(
                    generate_qa_candidate(
                        qa_model_id,
                        entity_fact.entity_name,
                        entity_fact.facts,
                        description,
                        complex_target,
                        root.carrier,
                    )
                    for complex_target in targets
                )
            )
            for candidate, error in outcomes:
                if error:
                    errors.append(error)
                if candidate is None:
                    continue
                key = " ".join(candidate.question.lower().split())
                if not key or key in seen_questions:
                    continue
                current_complex = sum(
                    qa.info.get("complexity") == "complex" for qa in accepted
                )
                remaining_complex = max(0, required_complex - current_complex)
                remaining_slots = target - len(accepted)
                if (
                    candidate.info.get("complexity") != "complex"
                    and remaining_slots <= remaining_complex
                ):
                    continue
                accepted.append(candidate)
                seen_questions.add(key)
                await append_output_record(paths.qas, qa_to_dict(candidate), paths.lock)
                if len(accepted) >= target:
                    break
        if len(accepted) < target:
            await append_output_record(
                paths.failures,
                {
                    "stage": "qa_generation_shortfall",
                    "entity": entity_fact.entity_name,
                    "requested": target,
                    "generated": len(accepted),
                    "candidate_errors": errors,
                },
                paths.lock,
            )
        root.set_output({"requested": target, "generated": len(accepted)})
        return EntityQASummary(
            entity_name=entity_fact.entity_name,
            requested=target,
            generated=len(accepted),
            phoenix_trace_id=root.trace_id,
        )


@flow(
    name="deep-search-tasks-generate-qas",
    flow_run_name="deep-search-tasks-generate-qas-{batch_output_dataset_name}",
    retries=0,
    persist_result=False,
)
async def generate_deep_search_qas_flow(
    prepare_run_directory: Path,
    batch_output_dataset_name: str,
    qa_pairs_per_entity: int = 4,
    qa_model_id: str = DEFAULT_QA_MODEL_ID,
    complex_pair_ratio: float = 0.7,
    max_qa_generation_attempts: int = 4,
    llm_concurrency: int = 25,
    download_timeout: float = 600.0,
) -> dict[str, Any]:
    qa_model_id = qa_model_id.strip()
    if qa_pairs_per_entity < 0:
        raise ValueError("qa_pairs_per_entity must be at least 0")
    if not qa_model_id:
        raise ValueError("qa_model_id must not be blank")
    if not 0.0 <= complex_pair_ratio <= 1.0:
        raise ValueError("complex_pair_ratio must be between 0 and 1")
    if max_qa_generation_attempts < 1:
        raise ValueError("max_qa_generation_attempts must be at least 1")
    if llm_concurrency < 1:
        raise ValueError("llm_concurrency must be at least 1")
    if download_timeout <= 0:
        raise ValueError("download_timeout must be greater than 0")
    prepare_run_directory = prepare_run_directory.expanduser().resolve()
    prepare_metadata_path = prepare_run_directory / "prepare_metadata.json"
    prepare_metadata = read_json(prepare_metadata_path)
    config = DeepSearchTaskGenerationConfig.model_validate(
        prepare_metadata.get("config")
    )
    run_id = str(flow_run.id)
    tracing = configure_tracing(project_name=phoenix_project())
    await _upsert_llm_concurrency_limit(llm_concurrency)
    paths = initialize_finalize_output(prepare_run_directory, run_id)
    finalize_config = {
        "qa_pairs_per_entity": qa_pairs_per_entity,
        "qa_model_id": qa_model_id,
        "complex_pair_ratio": complex_pair_ratio,
        "max_qa_generation_attempts": max_qa_generation_attempts,
        "llm_concurrency": llm_concurrency,
        "download_timeout": download_timeout,
    }
    stage = "download_batch_output"
    try:
        downloaded = await download_output(
            batch_output_dataset_name, paths.raw_directory, download_timeout
        )
        stage = "parse_batch_output"
        responses, entity_facts, diagnostics = await parse_fact_output(
            downloaded, prepare_run_directory / "fact_requests.jsonl"
        )
        write_jsonl(paths.entity_facts, (record.to_dict() for record in entity_facts))
        write_jsonl(paths.failures, diagnostics.failures)
        stage = "load_corpus"
        _, _, description = await load_corpus(config)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        append_jsonl(
            paths.failures,
            {"stage": stage, "error": error, "crashed": True},
            paths.lock,
        )
        failed_metadata = {
            "status": "failed",
            "config": config.model_dump(mode="json"),
            "finalize_config": finalize_config,
            "prefect_flow_run_id": run_id,
            "phoenix_project": tracing.project_name,
            "prepare_run_directory": str(prepare_run_directory),
            "prepare_prefect_flow_run_id": prepare_metadata.get("prefect_flow_run_id"),
            "fireworks": {
                "input_dataset_name": prepare_metadata.get("fireworks", {}).get(
                    "input_dataset_name"
                ),
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
        write_json(paths.metadata, failed_metadata)
        tracing.force_flush()
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
            append_jsonl(
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
    metadata = {
        "status": (
            "failed"
            if crashes
            else (
                "empty"
                if not downloaded or not responses or not entity_facts
                else "completed"
            )
        ),
        "config": config.model_dump(mode="json"),
        "finalize_config": finalize_config,
        "prefect_flow_run_id": run_id,
        "phoenix_project": tracing.project_name,
        "prepare_run_directory": str(prepare_run_directory),
        "prepare_prefect_flow_run_id": prepare_metadata.get("prefect_flow_run_id"),
        "fireworks": {
            "input_dataset_name": prepare_metadata.get("fireworks", {}).get(
                "input_dataset_name"
            ),
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
    write_json(paths.metadata, metadata)
    get_tracing(phoenix_project()).force_flush()
    if crashes:
        raise RuntimeError(
            f"{len(crashes)} entity QA flow(s) crashed; metadata written to "
            f"{paths.metadata}."
        )
    return metadata
