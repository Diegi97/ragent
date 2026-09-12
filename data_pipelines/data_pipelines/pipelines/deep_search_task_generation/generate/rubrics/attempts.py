import asyncio
import sys
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import anyio
from prefect import task

from data_pipelines.artifacts.append import append_jsonl
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AUDITS_DIRECTORY_ENV,
    DATA_SOURCE_ENV,
    EVALUATION_CONFIG_ENV,
    PYTHON_EXECUTABLE_ENV,
    SOLVER_MODEL_ENV,
    AuditExecutionError,
    SolverAudit,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.config import (
    EVALUATION_CONFIG,
    PiThinkingLevel,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.models import (
    FactWorkspace,
    QuestionRubricAssignment,
    QuestionRubricAttempt,
    RubricFinalizePaths,
    RubricGenerationResult,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.prompts import (
    QUESTION_RUBRIC_AGENT_SYSTEM_PROMPT,
    build_question_rubric_user_prompt,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.runtime import (
    run_pi,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.audits import (
    validate_question_rubric_audits,
)
from data_pipelines.pipelines.deep_search_task_generation.project import phoenix_project
from data_pipelines.tracing import (
    SpanKind,
    object_trace,
    set_span_output,
    stage_span,
)
from ragent_core.artifacts.question_rubric import QuestionRubricRecord


async def generate_question_rubrics(
    assignments: Sequence[QuestionRubricAssignment],
    *,
    model: str,
    solver_model: str,
    thinking: PiThinkingLevel | None,
    max_attempts: int,
    workspace: FactWorkspace,
    sessions_directory: Path,
    paths: RubricFinalizePaths,
    pi_concurrency: int,
    logger: Any,
) -> RubricGenerationResult:
    pending = list(assignments)
    accepted: dict[int, QuestionRubricRecord] = {}
    errors: dict[int, list[str]] = {item.slot: [] for item in assignments}
    phoenix_trace_ids: dict[int, str] = {}
    solver_audits: dict[int, SolverAudit] = {}
    seen_questions: set[str] = set()
    infrastructure_failures: dict[int, str] = {}
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
                phoenix_trace_ids[slot] = outcome.phoenix_trace_id
                if outcome.infrastructure_error:
                    infrastructure_failures[slot] = (
                        outcome.error or "PI execution failed"
                    )
                else:
                    infrastructure_failures.pop(slot, None)
                if outcome.record is None or outcome.solver_audit is None:
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
                solver_audits[slot] = outcome.solver_audit
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
    if infrastructure_failures:
        detail = "; ".join(
            f"slot {slot}: {error}"
            for slot, error in sorted(infrastructure_failures.items())
        )
        raise RuntimeError(
            f"Rubric execution failed after {max_attempts} attempts: {detail}"
        )
    return RubricGenerationResult(accepted, errors, phoenix_trace_ids, solver_audits)


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
    thinking: PiThinkingLevel | None,
    workspace: FactWorkspace,
    sessions_directory: Path,
) -> QuestionRubricAttempt:
    output_path = workspace.outputs_directory / assignment.filename
    # A retry must never accept the previous attempt's candidate by accident.
    await asyncio.to_thread(output_path.unlink, missing_ok=True)
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
        pi_completed = False
        try:
            with stage_span(
                root.carrier,
                "run_pi_question_rubric_agent",
                SpanKind.AGENT,
                {"prompt": prompt, "output_path": str(output_path)},
                {"llm.model_name": model},
                project_name=phoenix_project(),
            ) as span:
                await run_pi(
                    prompt=prompt,
                    model=model,
                    thinking=thinking,
                    working_directory=workspace.directory,
                    session_directory=sessions_directory,
                    session_name=(
                        f"question-rubric-{assignment.slot}-"
                        f"{assignment.entity_fact.entity_name}-attempt-{attempt}"
                    ),
                    system_instructions=QUESTION_RUBRIC_AGENT_SYSTEM_PROMPT,
                    environment={
                        PYTHON_EXECUTABLE_ENV: sys.executable,
                        EVALUATION_CONFIG_ENV: str(EVALUATION_CONFIG),
                        DATA_SOURCE_ENV: assignment.entity_fact.data_source,
                        SOLVER_MODEL_ENV: solver_model,
                        AUDITS_DIRECTORY_ENV: str(workspace.audits_directory),
                    },
                )
                pi_completed = True
                record = await anyio.to_thread.run_sync(
                    partial(
                        validate_question_rubric_file,
                        output_path,
                        allowed_doc_ids=workspace.allowed_doc_ids,
                        expected_entity=assignment.entity_fact.entity_name,
                    )
                )
                audits = await anyio.to_thread.run_sync(
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
                solver_audit=audits.solver,
                phoenix_trace_id=root.trace_id,
            )
        except Exception as exc:
            root.mark_error(exc)
            return QuestionRubricAttempt(
                assignment=assignment,
                error=f"{type(exc).__name__}: {exc}",
                infrastructure_error=not pi_completed
                or isinstance(exc, (OSError, AuditExecutionError)),
                phoenix_trace_id=root.trace_id,
            )
