import asyncio
import json
import tempfile
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import anyio
from prefect import flow, task
from prefect.client.orchestration import get_client
from prefect.concurrency.asyncio import concurrency
from prefect.runtime import flow_run

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    FactWorkspace,
    PiThinkingLevel,
    QuestionRubricAssignment,
    QuestionRubricAttempt,
    QuestionRubricRecord,
    RubricFinalizePaths,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.output import (
    copy_question_rubric_outputs,
    initialize_rubric_finalize_output,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.prompts import (
    QUESTION_RUBRIC_AGENT_SYSTEM_PROMPT,
    build_question_rubric_user_prompt,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.workspace import (
    create_fact_workspace,
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
from data_pipelines.tracing import (
    configure_tracing,
    get_tracing,
    object_trace,
    set_span_output,
    stage_span,
)

PI_CONCURRENCY_LIMIT = "deep-search-tasks-pi-agent"


def _validate_runtime_parameters(
    model: str,
    thinking: str | None,
    num_question_rubrics: int,
    pi_concurrency: int,
    max_attempts: int,
    download_timeout: float,
) -> tuple[str, str | None]:
    model = model.strip()
    if not model:
        raise ValueError("model must not be blank")
    if thinking is not None:
        thinking = thinking.strip().lower()
        allowed_thinking_levels = {level.value for level in PiThinkingLevel}
        if thinking not in allowed_thinking_levels:
            raise ValueError(
                "thinking must be one of: "
                + ", ".join(level.value for level in PiThinkingLevel)
            )
    if num_question_rubrics < 0:
        raise ValueError("num_question_rubrics must be at least 0")
    if pi_concurrency < 1:
        raise ValueError("pi_concurrency must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if download_timeout <= 0:
        raise ValueError("download_timeout must be greater than 0")
    return model, thinking


async def _upsert_pi_concurrency_limit(limit: int) -> None:
    async with get_client() as client:
        await client.upsert_global_concurrency_limit_by_name(
            PI_CONCURRENCY_LIMIT, limit
        )


def order_entity_facts(
    entity_facts: Sequence[EntityFactMemoryRecord], entities_path: Path
) -> list[EntityFactMemoryRecord]:
    records_by_name = {record.entity_name: record for record in entity_facts}
    ordered: list[EntityFactMemoryRecord] = []
    seen: set[str] = set()
    with entities_path.open(encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed entity record at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Entity record at line {line_number} is not a JSON object."
                )
            entity_name = str(value.get("name") or "").strip()
            record = records_by_name.get(entity_name)
            if record is not None and entity_name not in seen:
                ordered.append(record)
                seen.add(entity_name)
    ordered.extend(record for record in entity_facts if record.entity_name not in seen)
    return ordered


def build_question_rubric_assignments(
    entity_facts: Sequence[EntityFactMemoryRecord],
    num_question_rubrics: int,
) -> list[QuestionRubricAssignment]:
    if num_question_rubrics < 0:
        raise ValueError("num_question_rubrics must be at least 0")
    usable = [record for record in entity_facts if record.facts]
    if not usable:
        return []
    return [
        QuestionRubricAssignment(slot=slot, entity_fact=usable[slot % len(usable)])
        for slot in range(num_question_rubrics)
    ]


async def run_pi(
    *,
    prompt: str,
    model: str,
    thinking: str | None,
    folder: Path,
    system_instructions: str,
) -> None:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"PI working directory does not exist: {folder}")
    command = [
        "pi",
        "--print",
        "--no-session",
        "--model",
        model,
    ]
    if thinking is not None:
        command.extend(["--thinking", thinking])
    command.extend(
        [
            "--append-system-prompt",
            system_instructions,
            prompt,
        ]
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=folder,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"PI failed in {folder} with exit code {process.returncode}:\n"
            f"{stderr.decode(errors='replace')}"
        )


@task(
    name="generate-question-rubric-with-pi",
    task_run_name="question-rubric-{assignment.slot}-{assignment.entity_fact.entity_name}",
    retries=0,
    persist_result=False,
)
async def run_question_rubric_attempt(
    assignment: QuestionRubricAssignment,
    *,
    attempt: int,
    previous_errors: list[str],
    model: str,
    thinking: str | None,
    workspace: FactWorkspace,
) -> QuestionRubricAttempt:
    output_path = workspace.outputs_directory / assignment.filename
    output_path.unlink(missing_ok=True)
    prompt = build_question_rubric_user_prompt(
        assignment,
        attempt=attempt,
        previous_errors=previous_errors,
    )
    with object_trace(
        f"question-rubric-{assignment.slot}-{assignment.entity_fact.entity_name}",
        {
            "slot": assignment.slot,
            "entity": assignment.entity_fact.entity_name,
            "attempt": attempt,
        },
        {
            "entity.name": assignment.entity_fact.entity_name,
            "question_rubric.slot": assignment.slot,
            "question_rubric.attempt": attempt,
            "llm.model_name": model,
        },
        project_name=phoenix_project(),
    ) as root:
        try:
            with stage_span(
                root.carrier,
                "run_pi_question_rubric_agent",
                "AGENT",
                {"prompt": prompt, "output_path": str(output_path)},
                {"llm.model_name": model},
                project_name=phoenix_project(),
            ) as span:
                async with concurrency(PI_CONCURRENCY_LIMIT, strict=True):
                    await run_pi(
                        prompt=prompt,
                        model=model,
                        thinking=thinking,
                        folder=workspace.directory,
                        system_instructions=QUESTION_RUBRIC_AGENT_SYSTEM_PROMPT,
                    )
                record = await anyio.to_thread.run_sync(
                    partial(
                        validate_question_rubric_file,
                        output_path,
                        allowed_doc_ids=workspace.allowed_doc_ids,
                        expected_entity=assignment.entity_fact.entity_name,
                    )
                )
                set_span_output(span, record.model_dump(mode="json"))
            root.set_output(record.model_dump(mode="json"))
            return QuestionRubricAttempt(
                assignment=assignment,
                record=record,
                phoenix_trace_id=root.trace_id,
            )
        except Exception as exc:
            root.mark_error(exc)
            return QuestionRubricAttempt(
                assignment=assignment,
                error=f"{type(exc).__name__}: {exc}",
                phoenix_trace_id=root.trace_id,
            )


async def generate_question_rubrics(
    assignments: Sequence[QuestionRubricAssignment],
    *,
    model: str,
    thinking: str | None,
    max_attempts: int,
    workspace: FactWorkspace,
) -> tuple[
    dict[int, QuestionRubricRecord],
    dict[int, list[str]],
    dict[int, str],
]:
    pending = list(assignments)
    accepted: dict[int, QuestionRubricRecord] = {}
    errors: dict[int, list[str]] = {item.slot: [] for item in assignments}
    trace_ids: dict[int, str] = {}
    seen_questions: set[str] = set()
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        outcomes = await asyncio.gather(
            *(
                run_question_rubric_attempt(
                    assignment,
                    attempt=attempt,
                    previous_errors=errors[assignment.slot],
                    model=model,
                    thinking=thinking,
                    workspace=workspace,
                )
                for assignment in pending
            )
        )
        next_pending: list[QuestionRubricAssignment] = []
        for outcome in outcomes:
            slot = outcome.assignment.slot
            trace_ids[slot] = outcome.phoenix_trace_id
            if outcome.record is None:
                errors[slot].append(outcome.error or "PI produced no valid record.")
                next_pending.append(outcome.assignment)
                continue
            question_key = " ".join(outcome.record.question.lower().split())
            if question_key in seen_questions:
                errors[slot].append(
                    "Question duplicates an earlier accepted question-rubric record."
                )
                next_pending.append(outcome.assignment)
                continue
            seen_questions.add(question_key)
            accepted[slot] = outcome.record
        pending = next_pending
    return accepted, errors, trace_ids


def _entity_summaries(
    assignments: Sequence[QuestionRubricAssignment],
    accepted: dict[int, QuestionRubricRecord],
    trace_ids: dict[int, str],
) -> list[dict[str, Any]]:
    requested = Counter(item.entity_fact.entity_name for item in assignments)
    generated = Counter(
        assignment.entity_fact.entity_name
        for assignment in assignments
        if assignment.slot in accepted
    )
    traces: dict[str, list[str]] = {}
    for assignment in assignments:
        trace_id = trace_ids.get(assignment.slot)
        if trace_id:
            traces.setdefault(assignment.entity_fact.entity_name, []).append(trace_id)
    return [
        {
            "entity_name": entity_name,
            "requested": count,
            "generated": generated[entity_name],
            "phoenix_trace_ids": traces.get(entity_name, []),
        }
        for entity_name, count in requested.items()
    ]


def _paths_metadata(paths: RubricFinalizePaths) -> dict[str, str]:
    return {
        "rubric_finalize_run_directory": str(paths.directory),
        "raw_directory": str(paths.raw_directory),
        "outputs_directory": str(paths.outputs_directory),
        "entity_facts": str(paths.entity_facts),
        "question_rubrics": str(paths.question_rubrics),
        "failures": str(paths.failures),
    }


@flow(
    name="deep-search-tasks-generate-rubrics",
    flow_run_name="deep-search-tasks-generate-rubrics-{batch_output_dataset_name}",
    retries=0,
    persist_result=False,
)
async def generate_deep_search_rubrics_flow(
    prepare_run_directory: Path,
    batch_output_dataset_name: str,
    model: str,
    num_question_rubrics: int,
    thinking: str | None = None,
    pi_concurrency: int = 10,
    max_attempts: int = 4,
    download_timeout: float = 600.0,
) -> dict[str, Any]:
    model, thinking = _validate_runtime_parameters(
        model,
        thinking,
        num_question_rubrics,
        pi_concurrency,
        max_attempts,
        download_timeout,
    )
    prepare_run_directory = prepare_run_directory.expanduser().resolve()
    prepare_metadata = read_json(prepare_run_directory / "prepare_metadata.json")
    run_id = str(flow_run.id)
    tracing = configure_tracing(project_name=phoenix_project())
    paths = initialize_rubric_finalize_output(prepare_run_directory, run_id)
    stage = "configure_pi_concurrency"
    try:
        await _upsert_pi_concurrency_limit(pi_concurrency)
        stage = "download_batch_output"
        downloaded = await download_output(
            batch_output_dataset_name, paths.raw_directory, download_timeout
        )
        stage = "parse_batch_output"
        responses, entity_facts, diagnostics = await parse_fact_output(
            downloaded, prepare_run_directory / "fact_requests.jsonl"
        )
        ordered_entity_facts = order_entity_facts(
            entity_facts, prepare_run_directory / "entities.jsonl"
        )
        write_jsonl(
            paths.entity_facts,
            (record.to_dict() for record in ordered_entity_facts),
        )
        failures = list(diagnostics.failures)
        stage = "generate_question_rubrics"
        assignments = build_question_rubric_assignments(
            ordered_entity_facts, num_question_rubrics
        )
        accepted: dict[int, QuestionRubricRecord] = {}
        errors: dict[int, list[str]] = {}
        trace_ids: dict[int, str] = {}
        if num_question_rubrics and not assignments:
            failures.append(
                {
                    "stage": "question_rubric_generation",
                    "error": "No entities with extracted facts are available.",
                }
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix="entity_fact_question_rubric_"
            ) as temporary_directory:
                workspace = create_fact_workspace(
                    Path(temporary_directory), ordered_entity_facts
                )
                try:
                    accepted, errors, trace_ids = await generate_question_rubrics(
                        assignments,
                        model=model,
                        thinking=thinking,
                        max_attempts=max_attempts,
                        workspace=workspace,
                    )
                finally:
                    copy_question_rubric_outputs(
                        workspace.outputs_directory,
                        paths.outputs_directory,
                    )
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
        write_jsonl(
            paths.question_rubrics,
            (accepted[slot].model_dump(mode="json") for slot in sorted(accepted)),
        )
        write_jsonl(paths.failures, failures)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        append_jsonl(
            paths.failures,
            {"stage": stage, "error": error, "crashed": True},
            paths.lock,
        )
        write_json(
            paths.metadata,
            {
                "status": "failed",
                "prefect_flow_run_id": run_id,
                "phoenix_project": tracing.project_name,
                "prepare_run_directory": str(prepare_run_directory),
                "prepare_prefect_flow_run_id": prepare_metadata.get(
                    "prefect_flow_run_id"
                ),
                "rubric_finalize_config": {
                    "model": model,
                    "thinking": thinking,
                    "num_question_rubrics": num_question_rubrics,
                    "pi_concurrency": pi_concurrency,
                    "max_attempts": max_attempts,
                    "download_timeout": download_timeout,
                },
                "failed_stage": stage,
                "error": error,
                "paths": _paths_metadata(paths),
            },
        )
        tracing.force_flush()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise RuntimeError(
            f"Question-rubric generation failed during {stage}; metadata written "
            f"to {paths.metadata}."
        ) from exc

    generated_count = len(accepted)
    status = (
        "completed"
        if generated_count == num_question_rubrics
        else "empty"
        if generated_count == 0
        else "partial"
    )
    metadata = {
        "status": status,
        "prefect_flow_run_id": run_id,
        "phoenix_project": tracing.project_name,
        "prepare_run_directory": str(prepare_run_directory),
        "prepare_prefect_flow_run_id": prepare_metadata.get("prefect_flow_run_id"),
        "prepare_config": prepare_metadata.get("config"),
        "rubric_finalize_config": {
            "model": model,
            "thinking": thinking,
            "num_question_rubrics": num_question_rubrics,
            "pi_concurrency": pi_concurrency,
            "max_attempts": max_attempts,
            "download_timeout": download_timeout,
        },
        "fireworks": {
            "input_dataset_name": prepare_metadata.get("fireworks", {}).get(
                "input_dataset_name"
            ),
            "output_dataset_name": batch_output_dataset_name,
        },
        "downloaded_file_count": len(downloaded),
        "batch_response_count": len(responses),
        "entity_fact_count": len(ordered_entity_facts),
        "usable_entity_count": sum(
            bool(record.facts) for record in ordered_entity_facts
        ),
        "requested_question_rubric_count": num_question_rubrics,
        "question_rubric_count": generated_count,
        "failure_count": count_jsonl(paths.failures),
        "parse_diagnostics": diagnostics.to_dict(),
        "entity_summaries": _entity_summaries(assignments, accepted, trace_ids),
        "paths": _paths_metadata(paths),
    }
    write_json(paths.metadata, metadata)
    get_tracing(phoenix_project()).force_flush()
    return metadata
