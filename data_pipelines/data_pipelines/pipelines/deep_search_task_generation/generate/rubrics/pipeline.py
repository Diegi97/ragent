import asyncio
import json
import os
import random
import signal
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import anyio
from prefect import flow, get_run_logger, task
from prefect.runtime import flow_run

from data_pipelines.pipelines.deep_search_task_generation.config import (
    FACT_RESPONSES_RELATIVE_PATH,
    REPOSITORY_ROOT,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    DEFAULT_SOLVER_MODEL,
    FactWorkspace,
    PiThinkingLevel,
    QuestionRubricAssignment,
    QuestionRubricAttempt,
    QuestionRubricRecord,
    RubricFinalizePaths,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.output import (
    initialize_rubric_finalize_output,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.profile import (
    build_dataset_profile,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.prompts import (
    QUESTION_RUBRIC_AGENT_SYSTEM_PROMPT,
    build_question_rubric_user_prompt,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_audits,
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.workspace import (
    create_fact_workspace,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.shared import (
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

DEEP_SEARCH_PROJECT = REPOSITORY_ROOT / "environments/ragent_deep_search"
EVALUATION_CONFIG = DEEP_SEARCH_PROJECT / "evaluation.toml"
PI_CODING_AGENT_DIRECTORY_ENV = "PI_CODING_AGENT_DIR"
PI_PHOENIX_EXTENSION = Path("npm/node_modules/pi-phoenix/index.ts")
PI_TIMEOUT_SECONDS = 45 * 60


def _validate_runtime_parameters(
    model: str,
    solver_model: str,
    thinking: str | None,
    num_question_rubrics: int,
    pi_concurrency: int,
    max_attempts: int,
) -> tuple[str, str, str | None]:
    model = model.strip()
    if not model:
        raise ValueError("model must not be blank")
    solver_model = solver_model.strip()
    if not solver_model:
        raise ValueError("solver_model must not be blank")
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
    if not EVALUATION_CONFIG.is_file():
        raise FileNotFoundError(f"evaluation config not found: {EVALUATION_CONFIG}")
    return model, solver_model, thinking


def _pi_phoenix_extension() -> Path:
    configured_directory = os.getenv(PI_CODING_AGENT_DIRECTORY_ENV, "").strip()
    agent_directory = Path(configured_directory or "~/.pi/agent").expanduser()
    extension = (agent_directory / PI_PHOENIX_EXTENSION).resolve()
    if not extension.is_file():
        raise FileNotFoundError(
            f"pi-phoenix extension not found at {extension}; install it with "
            "'pi install npm:pi-phoenix' or set PI_CODING_AGENT_DIR"
        )
    return extension


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
    *,
    random_entities: bool = False,
    seed: int = 0,
) -> list[QuestionRubricAssignment]:
    if num_question_rubrics < 0:
        raise ValueError("num_question_rubrics must be at least 0")
    usable = [record for record in entity_facts if record.facts]
    if not usable:
        return []
    if random_entities:
        if num_question_rubrics > len(usable):
            raise ValueError(
                "random entity selection without replacement requires at least "
                f"{num_question_rubrics} usable entities; found {len(usable)}"
            )
        selected = random.Random(seed).sample(usable, k=num_question_rubrics)
        return [
            QuestionRubricAssignment(slot=slot, entity_fact=entity_fact)
            for slot, entity_fact in enumerate(selected)
        ]
    return [
        QuestionRubricAssignment(slot=slot, entity_fact=usable[slot % len(usable)])
        for slot in range(num_question_rubrics)
    ]


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def run_pi(
    *,
    prompt: str,
    model: str,
    thinking: str | None,
    folder: Path,
    session_directory: Path,
    session_name: str,
    system_instructions: str,
    environment: dict[str, str],
) -> None:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"PI working directory does not exist: {folder}")
    session_directory = session_directory.resolve()
    if not session_directory.is_dir():
        raise ValueError(f"PI session directory does not exist: {session_directory}")
    command = [
        "pi",
        "--print",
        "--model",
        model,
        "--session-dir",
        str(session_directory),
        "--name",
        session_name,
        "--no-extensions",
        "--extension",
        str(_pi_phoenix_extension()),
    ]
    if thinking is not None:
        command.extend(["--thinking", thinking])
    command.extend(
        [
            "--system-prompt",
            system_instructions,
            prompt,
        ]
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=folder,
        env={**os.environ, **environment},
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=PI_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        await _stop_process(process)
        raise TimeoutError(
            f"PI exceeded its {PI_TIMEOUT_SECONDS:g}-second timeout in {folder}"
        ) from exc
    except asyncio.CancelledError:
        await _stop_process(process)
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"PI failed in {folder} with exit code {process.returncode}:\n"
            f"{stderr.decode(errors='replace')}"
        )


@task(
    name="generate-question-rubric-with-pi",
    task_run_name=(
        "question-rubric-{assignment.slot}-{assignment.entity_fact.entity_name}"
        "-attempt-{attempt}"
    ),
    retries=0,
    persist_result=False,
)
async def run_question_rubric_attempt(
    assignment: QuestionRubricAssignment,
    *,
    attempt: int,
    previous_errors: list[str],
    model: str,
    solver_model: str,
    thinking: str | None,
    workspace: FactWorkspace,
    sessions_directory: Path,
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
                await run_pi(
                    prompt=prompt,
                    model=model,
                    thinking=thinking,
                    folder=workspace.directory,
                    session_directory=sessions_directory,
                    session_name=(
                        f"question-rubric-{assignment.slot}-"
                        f"{assignment.entity_fact.entity_name}-attempt-{attempt}"
                    ),
                    system_instructions=QUESTION_RUBRIC_AGENT_SYSTEM_PROMPT,
                    environment={
                        "RAGENT_PYTHON_EXECUTABLE": sys.executable,
                        "RAGENT_EVALUATION_CONFIG": str(EVALUATION_CONFIG),
                        "RAGENT_DATA_SOURCE": assignment.entity_fact.data_source,
                        "RAGENT_SOLVER_MODEL": solver_model,
                        "RAGENT_AUDITS_DIRECTORY": str(workspace.audits_directory),
                    },
                )
                record = await anyio.to_thread.run_sync(
                    partial(
                        validate_question_rubric_file,
                        output_path,
                        allowed_doc_ids=workspace.allowed_doc_ids,
                        expected_entity=assignment.entity_fact.entity_name,
                    )
                )
                await anyio.to_thread.run_sync(
                    validate_question_rubric_audits,
                    output_path,
                    workspace.audits_directory,
                    record,
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
    solver_model: str,
    thinking: str | None,
    max_attempts: int,
    workspace: FactWorkspace,
    sessions_directory: Path,
    paths: RubricFinalizePaths,
    pi_concurrency: int,
    logger: Any,
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
    semaphore = asyncio.Semaphore(pi_concurrency)

    async def run_bounded(
        assignment: QuestionRubricAssignment, attempt: int
    ) -> QuestionRubricAttempt:
        async with semaphore:
            return await run_question_rubric_attempt(
                assignment,
                attempt=attempt,
                previous_errors=errors[assignment.slot],
                model=model,
                solver_model=solver_model,
                thinking=thinking,
                workspace=workspace,
                sessions_directory=sessions_directory,
            )

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        logger.info(
            "Starting rubric attempt round: attempt=%s pending=%s accepted=%s",
            attempt,
            len(pending),
            len(accepted),
        )
        next_pending: list[QuestionRubricAssignment] = []
        tasks = [
            asyncio.create_task(run_bounded(assignment, attempt))
            for assignment in pending
        ]
        try:
            for completed in asyncio.as_completed(tasks):
                outcome = await completed
                slot = outcome.assignment.slot
                entity = outcome.assignment.entity_fact.entity_name
                trace_ids[slot] = outcome.phoenix_trace_id
                if outcome.record is None:
                    error = outcome.error or "PI produced no valid record."
                    errors[slot].append(error)
                    next_pending.append(outcome.assignment)
                    logger.warning(
                        "Rubric attempt failed: slot=%s attempt=%s entity=%r "
                        "result=error accepted=%s error=%s",
                        slot,
                        attempt,
                        entity,
                        len(accepted),
                        error,
                    )
                    continue
                question_key = " ".join(outcome.record.question.lower().split())
                if question_key in seen_questions:
                    error = (
                        "Question duplicates an earlier accepted question-rubric "
                        "record."
                    )
                    errors[slot].append(error)
                    next_pending.append(outcome.assignment)
                    logger.warning(
                        "Rubric attempt rejected: slot=%s attempt=%s entity=%r "
                        "result=duplicate accepted=%s",
                        slot,
                        attempt,
                        entity,
                        len(accepted),
                    )
                    continue
                await anyio.to_thread.run_sync(
                    append_jsonl,
                    paths.question_rubrics,
                    outcome.record.model_dump(mode="json"),
                    paths.lock,
                )
                seen_questions.add(question_key)
                accepted[slot] = outcome.record
                logger.info(
                    "Rubric accepted: slot=%s attempt=%s entity=%r "
                    "result=accepted accepted=%s",
                    slot,
                    attempt,
                    entity,
                    len(accepted),
                )
        finally:
            for running_task in tasks:
                if not running_task.done():
                    running_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        pending = next_pending
        if pending and attempt < max_attempts:
            delay_seconds = min(2 ** (attempt - 1), 10)
            logger.info(
                "Waiting before next rubric attempt round: attempt=%s "
                "pending=%s delay_seconds=%s",
                attempt + 1,
                len(pending),
                delay_seconds,
            )
            await asyncio.sleep(delay_seconds)
    logger.info(
        "Rubric generation finished: requested=%s accepted=%s shortfall=%s",
        len(assignments),
        len(accepted),
        len(assignments) - len(accepted),
    )
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
        "workspace_directory": str(paths.workspace_directory),
        "outputs_directory": str(paths.outputs_directory),
        "sessions_directory": str(paths.sessions_directory),
        "entity_facts": str(paths.entity_facts),
        "question_rubrics": str(paths.question_rubrics),
        "failures": str(paths.failures),
    }


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
    solver_model: str = DEFAULT_SOLVER_MODEL,
    thinking: str | None = None,
    pi_concurrency: int = 10,
    max_attempts: int = 4,
    random_entities: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    model, solver_model, thinking = _validate_runtime_parameters(
        model,
        solver_model,
        thinking,
        num_question_rubrics,
        pi_concurrency,
        max_attempts,
    )
    prepare_run_directory = prepare_run_directory.expanduser().resolve()
    prepare_metadata = read_json(prepare_run_directory / "prepare_metadata.json")
    fact_responses_path = prepare_run_directory / FACT_RESPONSES_RELATIVE_PATH
    if not fact_responses_path.is_file():
        raise FileNotFoundError(
            f"Fireworks fact responses not found: {fact_responses_path}. Place "
            f"{FACT_RESPONSES_RELATIVE_PATH.name} at this path before running "
            "generate-rubrics."
        )
    run_id = str(flow_run.id)
    tracing = configure_tracing(project_name=phoenix_project())
    logger = get_run_logger()
    paths = initialize_rubric_finalize_output(prepare_run_directory, run_id)
    stage = "parse_batch_output"
    try:
        responses, entity_facts, diagnostics = await parse_fact_output(
            [fact_responses_path], prepare_run_directory / "fact_requests.jsonl"
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
            ordered_entity_facts,
            num_question_rubrics,
            random_entities=random_entities,
            seed=seed,
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
            workspace = create_fact_workspace(
                paths.workspace_directory, ordered_entity_facts
            )
            accepted, errors, trace_ids = await generate_question_rubrics(
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
                    "solver_model": solver_model,
                    "thinking": thinking,
                    "num_question_rubrics": num_question_rubrics,
                    "pi_concurrency": pi_concurrency,
                    "max_attempts": max_attempts,
                    "random_entities": random_entities,
                    "seed": seed,
                },
                "fireworks": {
                    "input_dataset_name": prepare_metadata.get("fireworks", {}).get(
                        "input_dataset_name"
                    ),
                    "output_file": str(fact_responses_path),
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
            "solver_model": solver_model,
            "thinking": thinking,
            "num_question_rubrics": num_question_rubrics,
            "pi_concurrency": pi_concurrency,
            "max_attempts": max_attempts,
            "random_entities": random_entities,
            "seed": seed,
        },
        "fireworks": {
            "input_dataset_name": prepare_metadata.get("fireworks", {}).get(
                "input_dataset_name"
            ),
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
        "entity_summaries": _entity_summaries(assignments, accepted, trace_ids),
        "dataset_profile": build_dataset_profile(
            assignments,
            accepted,
            paths.workspace_directory / ".difficulty_checks",
        ),
        "paths": _paths_metadata(paths),
    }
    write_json(paths.metadata, metadata)
    get_tracing(phoenix_project()).force_flush()
    return metadata
