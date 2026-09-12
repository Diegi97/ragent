import asyncio
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from prefect import flow, get_run_logger
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
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.assignments import (
    build_question_rubric_assignments,
    order_entity_facts,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.attempts import (
    generate_question_rubrics,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    SolverAudit,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.config import (
    EVALUATION_CONFIG,
    PiThinkingLevel,
    RubricGenerationConfig,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    QuestionRubricAssignment,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.output import (
    initialize_rubric_finalize_output,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.profile import (
    build_dataset_profile,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.workspace import (
    create_fact_workspace,
)
from data_pipelines.pipelines.deep_search_task_generation.prepare.models import (
    PreparePaths,
    PrepareRunMetadata,
)
from data_pipelines.pipelines.deep_search_task_generation.project import phoenix_project
from data_pipelines.tracing import (
    configure_tracing,
)
from ragent_core.artifacts.question_rubric import (
    QuestionRubricDatasetMetadata,
    QuestionRubricRecord,
)


@flow(
    name="deep-search-tasks-generate-rubrics",
    flow_run_name="deep-search-tasks-generate-rubrics",
    retries=0,
    persist_result=False,
)
async def generate_deep_search_rubrics_flow(
    prepare_run_directory: Path,
    model: str,
    num_question_rubrics: int,
    solver_model: str = RubricGenerationConfig.model_fields["solver_model"].default,
    thinking: PiThinkingLevel | None = RubricGenerationConfig.model_fields[
        "thinking"
    ].default,
    pi_concurrency: int = RubricGenerationConfig.model_fields["pi_concurrency"].default,
    max_attempts: int = RubricGenerationConfig.model_fields["max_attempts"].default,
    random_entities: bool = RubricGenerationConfig.model_fields[
        "random_entities"
    ].default,
    seed: int = RubricGenerationConfig.model_fields["seed"].default,
) -> dict[str, Any]:
    generation_config = RubricGenerationConfig(
        model=model,
        solver_model=solver_model,
        thinking=thinking,
        num_question_rubrics=num_question_rubrics,
        pi_concurrency=pi_concurrency,
        max_attempts=max_attempts,
        random_entities=random_entities,
        seed=seed,
    )
    model = generation_config.model
    solver_model = generation_config.solver_model
    thinking = generation_config.thinking
    if not EVALUATION_CONFIG.is_file():
        raise FileNotFoundError(f"evaluation config not found: {EVALUATION_CONFIG}")
    prepare_run_directory = prepare_run_directory.expanduser().resolve()
    prepare_paths = PreparePaths.in_directory(prepare_run_directory)
    prepare_metadata = PrepareRunMetadata.model_validate(
        await asyncio.to_thread(read_json, prepare_paths.metadata)
    )
    fact_responses_path = prepare_paths.fact_responses
    if not fact_responses_path.is_file():
        raise FileNotFoundError(
            f"Fireworks fact responses not found: {fact_responses_path}. Place "
            f"{prepare_paths.fact_responses.name} at this path before running "
            "generate-rubrics."
        )
    run_id = str(flow_run.id)
    tracing = configure_tracing(project_name=phoenix_project())
    logger = get_run_logger()
    paths = await asyncio.to_thread(
        initialize_rubric_finalize_output, prepare_run_directory, run_id
    )
    stage = "parse_batch_output"
    try:
        responses, entity_facts, diagnostics = await parse_fact_output(
            [fact_responses_path], prepare_paths.fact_requests
        )
        ordered_entity_facts = await asyncio.to_thread(
            order_entity_facts, entity_facts, prepare_paths.entities
        )
        await asyncio.to_thread(
            write_jsonl,
            paths.entity_facts,
            (record.to_dict() for record in ordered_entity_facts),
        )
        failures = list(diagnostics.failures)
        stage = "generate_question_rubrics"
        assignments = build_question_rubric_assignments(
            ordered_entity_facts,
            generation_config,
        )
        logger.info(
            "Rubric assignments ready: requested=%s assigned=%s "
            "random_entities=%s seed=%s pi_concurrency=%s",
            num_question_rubrics,
            len(assignments),
            random_entities,
            seed,
            pi_concurrency,
        )
        accepted: dict[int, QuestionRubricRecord] = {}
        solver_audits: dict[int, SolverAudit] = {}
        errors: dict[int, list[str]] = {}
        phoenix_trace_ids: dict[int, str] = {}
        if num_question_rubrics and not assignments:
            failures.append(
                {
                    "stage": "question_rubric_generation",
                    "error": "No entities with extracted facts are available.",
                }
            )
        else:
            workspace = await asyncio.to_thread(
                create_fact_workspace, paths.workspace_directory, ordered_entity_facts
            )
            generation_result = await generate_question_rubrics(
                assignments,
                model=model,
                solver_model=solver_model,
                thinking=thinking,
                max_attempts=max_attempts,
                workspace=workspace,
                sessions_directory=paths.sessions_directory,
                paths=paths,
                pi_concurrency=pi_concurrency,
                logger=logger,
            )
            accepted = generation_result.accepted
            errors = generation_result.errors
            phoenix_trace_ids = generation_result.phoenix_trace_ids
            solver_audits = generation_result.solver_audits
        for assignment in assignments:
            if assignment.slot not in accepted:
                failures.append(
                    {
                        "stage": "question_rubric_generation_shortfall",
                        "slot": assignment.slot,
                        "entity": assignment.entity_fact.entity_name,
                        "attempts": max_attempts,
                        "errors": errors.get(assignment.slot, []),
                    }
                )
        await asyncio.to_thread(
            write_jsonl,
            paths.question_rubrics,
            (accepted[slot].model_dump(mode="json") for slot in sorted(accepted)),
        )
        await asyncio.to_thread(write_jsonl, paths.failures, failures)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        await asyncio.to_thread(
            append_jsonl,
            paths.failures,
            {"stage": stage, "error": error, "crashed": True},
            paths.lock,
        )
        await asyncio.to_thread(
            write_json,
            paths.metadata,
            {
                "status": GenerationStatus.FAILED,
                "prefect_flow_run_id": run_id,
                "phoenix_project": tracing.project_name,
                "prepare_run_directory": str(prepare_run_directory),
                "prepare_prefect_flow_run_id": prepare_metadata.prefect_flow_run_id,
                "rubric_finalize_config": {
                    "model": model,
                    "solver_model": solver_model,
                    "thinking": thinking,
                    "num_question_rubrics": num_question_rubrics,
                    "pi_concurrency": pi_concurrency,
                    "max_attempts": max_attempts,
                    "random_entities": random_entities,
                    "seed": seed,
                },
                "fireworks": {
                    "input_dataset_name": prepare_metadata.fireworks.input_dataset_name,
                    "output_file": str(fact_responses_path),
                },
                "failed_stage": stage,
                "error": error,
                "paths": paths.to_metadata(),
            },
        )
        await asyncio.to_thread(tracing.force_flush)
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise RuntimeError(
            f"Question-rubric generation failed during {stage}; metadata written "
            f"to {paths.metadata}."
        ) from exc

    generated_count = len(accepted)
    status = GenerationStatus.for_output(
        generated_count, num_question_rubrics, diagnostics.has_integrity_errors
    )
    metadata = {
        "status": status,
        "prefect_flow_run_id": run_id,
        "phoenix_project": tracing.project_name,
        "prepare_run_directory": str(prepare_run_directory),
        "prepare_prefect_flow_run_id": prepare_metadata.prefect_flow_run_id,
        "prepare_config": prepare_metadata.config.model_dump(mode="json"),
        "rubric_finalize_config": {
            "model": model,
            "solver_model": solver_model,
            "thinking": thinking,
            "num_question_rubrics": num_question_rubrics,
            "pi_concurrency": pi_concurrency,
            "max_attempts": max_attempts,
            "random_entities": random_entities,
            "seed": seed,
        },
        "fireworks": {
            "input_dataset_name": prepare_metadata.fireworks.input_dataset_name,
            "output_file": str(fact_responses_path),
        },
        "batch_response_count": len(responses),
        "entity_fact_count": len(ordered_entity_facts),
        "usable_entity_count": sum(
            bool(record.facts) for record in ordered_entity_facts
        ),
        "requested_question_rubric_count": num_question_rubrics,
        "question_rubric_count": generated_count,
        "failure_count": count_jsonl(paths.failures),
        "parse_diagnostics": diagnostics.to_dict(),
        "entity_summaries": _entity_summaries(assignments, accepted, phoenix_trace_ids),
        "dataset_profile": await asyncio.to_thread(
            build_dataset_profile,
            accepted,
            solver_audits,
        ),
        "paths": paths.to_metadata(),
    }
    metadata = QuestionRubricDatasetMetadata.model_validate(metadata).model_dump(
        mode="json", exclude_unset=True
    )
    await asyncio.to_thread(write_json, paths.metadata, metadata)
    await asyncio.to_thread(
        configure_tracing(project_name=phoenix_project()).force_flush
    )
    return metadata


def _entity_summaries(
    assignments: Sequence[QuestionRubricAssignment],
    accepted: dict[int, QuestionRubricRecord],
    phoenix_trace_ids: dict[int, str],
) -> list[dict[str, Any]]:
    requested = Counter(item.entity_fact.entity_name for item in assignments)
    generated = Counter(
        assignment.entity_fact.entity_name
        for assignment in assignments
        if assignment.slot in accepted
    )
    traces: dict[str, list[str]] = {}
    for assignment in assignments:
        phoenix_trace_id = phoenix_trace_ids.get(assignment.slot)
        if phoenix_trace_id:
            traces.setdefault(assignment.entity_fact.entity_name, []).append(
                phoenix_trace_id
            )
    return [
        {
            "entity_name": entity_name,
            "requested": count,
            "generated": generated[entity_name],
            "phoenix_trace_ids": traces.get(entity_name, []),
        }
        for entity_name, count in requested.items()
    ]
