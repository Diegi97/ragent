import asyncio
import logging
import time
from typing import Any

import verifiers.v1 as vf

from ragent_core.judges.evaluation.models import (
    SCHEMA_VERSION,
    BatchJob,
    CallResult,
    GroundTruthExample,
)
from ragent_core.judges.rubric import RubricJudge
from ragent_core.judges.rubric.config import RubricJudgeConfig
from ragent_core.judges.rubric.contracts import (
    JudgeCriterion,
)

logger = logging.getLogger(__name__)


async def run_experiments(
    examples: list[GroundTruthExample],
    criteria_per_call: list[int],
    *,
    judge_model: str,
    base_url: str,
    api_key_var: str,
    temperature: float | None,
    max_tokens: int,
    max_concurrent: int,
) -> tuple[list[dict[str, Any]], list[CallResult]]:
    semaphore = asyncio.Semaphore(max_concurrent)
    all_rows: list[dict[str, Any]] = []
    all_calls: list[CallResult] = []
    for requested_size in criteria_per_call:
        sampling: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            sampling["temperature"] = temperature
        judge = RubricJudge(
            RubricJudgeConfig(
                model=judge_model,
                base_url=base_url,
                api_key_var=api_key_var,
                max_criteria=requested_size,
                sampling=sampling,
            )
        )
        jobs = _jobs_for_size(examples, requested_size)
        logger.info(
            "Judging %d criteria in %d calls with criteria_per_call=%d",
            sum(len(job.criteria) for job in jobs),
            len(jobs),
            requested_size,
        )
        results = await asyncio.gather(
            *(_grade_job(job, judge, semaphore) for job in jobs)
        )
        all_calls.extend(results)
        all_rows.extend(row for result in results for row in result.rows)
        failures = sum(result.error is not None for result in results)
        logger.info(
            "Completed criteria_per_call=%d: %d/%d calls parseable",
            requested_size,
            len(results) - failures,
            len(results),
        )
    return all_rows, all_calls


def _make_trace(job: BatchJob) -> vf.Trace:
    task_data = vf.TaskData(
        idx=job.task_index,
        prompt=job.example.question,
    )
    return vf.Trace(
        task=vf.TraceTask(type="RubricJudgeEvaluation", data=task_data),
        # This synthetic trace regrades an existing response without running an agent.
        agent=vf.AgentInfo(
            config=vf.AgentConfig(), name="recorded-response", trainable=False
        ),
        nodes=[
            vf.MessageNode(
                message=vf.AssistantMessage(content=job.example.response),
                sampled=True,
            )
        ],
    )


def _response_details(trace: vf.Trace) -> tuple[str | None, dict[str, dict[str, str]]]:
    records = trace.info.get("judge")
    if not isinstance(records, list) or not records:
        return None, {}
    record = records[-1]
    if not isinstance(record, dict):
        return None, {}
    raw_response = record.get("text")
    parsed = record.get("parsed")
    by_id: dict[str, dict[str, str]] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            by_id[item["id"]] = {
                "reason": str(item.get("reason", "")),
                "verdict": str(item.get("verdict", "")),
            }
    return raw_response if isinstance(raw_response, str) else None, by_id


def _usage_details(trace: vf.Trace) -> dict[str, float | int | None] | None:
    usage = vf.Usage.aggregate(trace.extra_usage)
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "input_tokens": usage.input_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cost": usage.cost,
    }


async def _grade_job(
    job: BatchJob,
    judge: RubricJudge,
    semaphore: asyncio.Semaphore,
) -> CallResult:
    trace = _make_trace(job)
    batch = [JudgeCriterion(id=item.id, text=item.text) for item in job.criteria]
    scores: dict[str, float] = {}
    error: str | None = None
    started = 0.0
    async with semaphore:
        started = time.perf_counter()
        try:
            scores = await judge.grade_batch(
                trace=trace,
                question=job.example.question,
                response=job.example.response,
                batch=batch,
            )
        except Exception as exc:  # Judge noncompliance is an experimental result.
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

    raw_response, verdicts = _response_details(trace)
    call_id = (
        f"k{job.requested_size}:{job.example.example_id}:batch{job.batch_index:03d}"
    )
    rows: list[dict[str, Any]] = []
    for position, criterion in enumerate(job.criteria, start=1):
        prediction_value = scores.get(criterion.id) if error is None else None
        prediction = (
            int(prediction_value) if prediction_value in (0, 0.0, 1, 1.0) else None
        )
        verdict = verdicts.get(criterion.id, {})
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "call_id": call_id,
                "example_id": job.example.example_id,
                "criterion_id": criterion.id,
                "criterion_index": criterion.index,
                "criterion": criterion.text,
                "ground_truth": criterion.score,
                "prediction": prediction,
                "is_correct": prediction == criterion.score
                if prediction is not None
                else False,
                "judge_reason": verdict.get("reason"),
                "judge_verdict": verdict.get("verdict"),
                "judge_model": judge.config.model,
                "judge_response": raw_response,
                "error": error,
                "criteria_per_call": job.requested_size,
                "actual_criteria_in_call": len(job.criteria),
                "batch_index": job.batch_index,
                "position_in_call": position,
                "elapsed_seconds": elapsed,
            }
        )

    return CallResult(
        call_id=call_id,
        example_id=job.example.example_id,
        requested_size=job.requested_size,
        actual_size=len(job.criteria),
        batch_index=job.batch_index,
        elapsed_seconds=elapsed,
        error=error,
        raw_response=raw_response,
        usage=_usage_details(trace),
        rows=rows,
    )


def _jobs_for_size(
    examples: list[GroundTruthExample],
    requested_size: int,
) -> list[BatchJob]:
    jobs: list[BatchJob] = []
    for task_index, example in enumerate(examples):
        for batch_index, start in enumerate(
            range(0, len(example.criteria), requested_size),
            start=1,
        ):
            jobs.append(
                BatchJob(
                    example=example,
                    task_index=task_index,
                    requested_size=requested_size,
                    batch_index=batch_index,
                    criteria=example.criteria[start : start + requested_size],
                )
            )
    return jobs
