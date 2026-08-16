"""Evaluate batched rubric judgments against criterion-level teacher labels.

First run the deep-search evaluation configured by
``environments/ragent_deep_search/rubric_judge_ground_truth_generation.toml``.
It uses the null harness to generate retrieval-assisted answers and configures the
teacher judge with ``max_criteria = 1``, producing one independent teacher call per
rubric criterion. The resulting Verifiers ``traces.jsonl`` is the ground-truth input
to this script; no intermediate conversion or ground-truth generation script is
needed. A run output directory may be passed in place of the traces file.

For every requested ``--criteria-per-call`` value, this script rejudges the saved
answers with the repository's production ``RubricJudge`` prompt and parser. It writes
detailed metrics and criterion-level predictions while printing only a compact
aggregate summary. Malformed or incomplete judge calls remain experimental failures
and count as incorrect in strict accuracy.

Example:
    uv run --env-file environments/ragent_deep_search/.env \
        --project ragent_core python \
        ragent_core/scripts/evaluate_rubric_judge.py \
        environments/ragent_deep_search/outputs/<run>/traces.jsonl \
        openai/gpt-5.4-nano --criteria-per-call 1 2 4 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from verifiers.v1.judges.rubric import Criterion

from ragent_core.judges.rubric import RubricJudge, RubricJudgeConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_API_KEY_VAR = "PRIME_API_KEY"


@dataclass(frozen=True)
class GroundTruthCriterion:
    name: str
    text: str
    score: int
    index: int


@dataclass(frozen=True)
class GroundTruthExample:
    example_id: str
    question: str
    response: str
    criteria: tuple[GroundTruthCriterion, ...]


@dataclass(frozen=True)
class BatchJob:
    example: GroundTruthExample
    task_index: int
    requested_size: int
    batch_index: int
    criteria: tuple[GroundTruthCriterion, ...]


@dataclass
class CallResult:
    call_id: str
    example_id: str
    requested_size: int
    actual_size: int
    batch_index: int
    elapsed_seconds: float
    error: str | None
    raw_response: str | None
    usage: dict[str, float | int | None] | None
    rows: list[dict[str, Any]]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ground_truth_path",
        type=Path,
        help="Teacher-labeled Verifiers traces.jsonl file or its run directory.",
    )
    parser.add_argument("judge_model")
    parser.add_argument(
        "--criteria-per-call",
        type=_positive_int,
        nargs="+",
        required=True,
        help="One or more maximum rubric batch sizes to compare.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-var", default=DEFAULT_API_KEY_VAR)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=_positive_int, default=4096)
    parser.add_argument(
        "--max-concurrent",
        type=_positive_int,
        default=16,
        help="Maximum in-flight judge calls across examples.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Metrics JSON path. Defaults next to the input file.",
    )
    parser.add_argument(
        "--judgments-output",
        type=Path,
        help="Criterion-level JSONL path. Defaults next to the metrics file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing metrics and judgment files.",
    )
    return parser.parse_args()


def _resolve_traces_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "traces.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Teacher-labeled traces file not found: {path}")
    return path


def _trace_response(trace: dict[str, Any]) -> str:
    sampled_messages = [
        node.get("message", {})
        for node in trace.get("nodes", [])
        if isinstance(node, dict)
        and node.get("sampled")
        and isinstance(node.get("message"), dict)
        and node["message"].get("role") == "assistant"
    ]
    if not sampled_messages:
        raise ValueError("trace contains no sampled assistant messages")
    response = sampled_messages[-1].get("content")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("trace's final sampled assistant message has no text")
    return response.strip()


def _teacher_scores(
    trace: dict[str, Any],
    expected_names: set[str],
) -> dict[str, int]:
    info = trace.get("info")
    calls = info.get("judge") if isinstance(info, dict) else None
    if not isinstance(calls, list) or len(calls) != len(expected_names):
        raise ValueError(
            "trace must contain exactly one teacher judge call per rubric criterion"
        )

    scores: dict[str, int] = {}
    for call_index, call in enumerate(calls, start=1):
        parsed = call.get("parsed") if isinstance(call, dict) else None
        if not isinstance(parsed, list) or len(parsed) != 1:
            raise ValueError(
                f"teacher call {call_index} did not grade exactly one criterion"
            )
        verdict = parsed[0]
        if not isinstance(verdict, dict):
            raise ValueError(f"teacher call {call_index} has an invalid verdict")
        name = verdict.get("name")
        answer = verdict.get("verdict")
        if not isinstance(name, str) or name not in expected_names:
            raise ValueError(
                f"teacher call {call_index} returned unexpected criterion {name!r}"
            )
        if name in scores:
            raise ValueError(f"teacher returned duplicate verdict for {name!r}")
        if not isinstance(answer, str):
            raise ValueError(f"teacher call {call_index} has no verdict")
        normalized = answer.strip().casefold()
        if normalized == "yes":
            scores[name] = 1
        elif normalized == "no":
            scores[name] = 0
        else:
            raise ValueError(
                f"teacher returned {answer!r} for {name}; expected yes or no"
            )

    if set(scores) != expected_names:
        raise ValueError("teacher judgments do not cover every rubric criterion")
    return scores


def _example_from_trace(trace: dict[str, Any]) -> GroundTruthExample:
    if trace.get("errors"):
        raise ValueError("trace contains rollout or teacher-scoring errors")
    example_id = trace.get("id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError("trace has no ID")

    task = trace.get("task")
    data = task.get("data") if isinstance(task, dict) else None
    if not isinstance(data, dict):
        raise ValueError("trace contains no task.data object")
    question = data.get("question", data.get("prompt"))
    raw_criteria = data.get("rubric")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("trace task contains no question")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise ValueError("trace task contains no rubric criteria")

    criterion_items: list[tuple[str, str, int]] = []
    for criterion_index, raw_criterion in enumerate(raw_criteria, start=1):
        text = (
            raw_criterion.get("criterion") if isinstance(raw_criterion, dict) else None
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"rubric criterion {criterion_index} has no text")
        criterion_items.append(
            (f"criterion_{criterion_index:02d}", text.strip(), criterion_index)
        )

    expected_names = {name for name, _, _ in criterion_items}
    teacher_scores = _teacher_scores(trace, expected_names)
    metrics = trace.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("trace contains no criterion metrics")

    criteria: list[GroundTruthCriterion] = []
    for name, text, criterion_index in criterion_items:
        metric_name = f"rubric/{name}"
        metric = metrics.get(metric_name)
        if metric not in (0, 0.0, 1, 1.0):
            raise ValueError(f"trace contains no binary {metric_name!r} metric")
        score = teacher_scores[name]
        if int(metric) != score:
            raise ValueError(f"teacher verdict and {metric_name!r} disagree")
        criteria.append(
            GroundTruthCriterion(
                name=name,
                text=text,
                score=score,
                index=criterion_index,
            )
        )

    return GroundTruthExample(
        example_id=example_id,
        question=question.strip(),
        response=_trace_response(trace),
        criteria=tuple(criteria),
    )


def _load_ground_truth(path: Path) -> list[GroundTruthExample]:
    examples: list[GroundTruthExample] = []
    seen_example_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc.msg}"
                ) from exc
            if not isinstance(trace, dict):
                raise ValueError(f"Line {line_number} of {path} must be a JSON object")
            try:
                example = _example_from_trace(trace)
            except ValueError as exc:
                raise ValueError(f"Invalid trace on line {line_number}: {exc}") from exc
            if example.example_id in seen_example_ids:
                raise ValueError(
                    f"Duplicate trace ID {example.example_id!r} on line {line_number}"
                )
            seen_example_ids.add(example.example_id)
            examples.append(example)

    if not examples:
        raise ValueError(f"Teacher-labeled traces file is empty: {path}")
    return examples


def _make_trace(job: BatchJob) -> vf.Trace:
    task_data = vf.TaskData(
        idx=job.task_index,
        prompt=job.example.question,
    )
    return vf.Trace(
        task=vf.TraceTask(type="RubricJudgeEvaluation", data=task_data),
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
    by_name: dict[str, dict[str, str]] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            by_name[item["name"]] = {
                "reason": str(item.get("reason", "")),
                "verdict": str(item.get("verdict", "")),
            }
    return raw_response if isinstance(raw_response, str) else None, by_name


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
    batch = [Criterion(name=item.name, text=item.text) for item in job.criteria]
    scores: dict[str, float] = {}
    error: str | None = None
    started = 0.0
    async with semaphore:
        started = time.perf_counter()
        try:
            scores = await judge._grade_batch(
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
        prediction_value = scores.get(criterion.name) if error is None else None
        prediction = (
            int(prediction_value) if prediction_value in (0, 0.0, 1, 1.0) else None
        )
        verdict = verdicts.get(criterion.name, {})
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "call_id": call_id,
                "example_id": job.example.example_id,
                "criterion_name": criterion.name,
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


async def _run_experiments(
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


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    return numerator / denominator if denominator else None


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    graded = [row for row in rows if row["prediction"] is not None]
    correct = sum(row["prediction"] == row["ground_truth"] for row in graded)
    tp = sum(row["prediction"] == 1 and row["ground_truth"] == 1 for row in graded)
    tn = sum(row["prediction"] == 0 and row["ground_truth"] == 0 for row in graded)
    fp = sum(row["prediction"] == 1 and row["ground_truth"] == 0 for row in graded)
    fn = sum(row["prediction"] == 0 and row["ground_truth"] == 1 for row in graded)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else None

    count = len(graded)
    observed_agreement = _ratio(correct, count)
    expected_agreement = None
    cohen_kappa = None
    if count:
        expected_agreement = ((tp + fp) / count) * ((tp + fn) / count) + (
            (tn + fn) / count
        ) * ((tn + fp) / count)
        if observed_agreement is not None and expected_agreement != 1:
            cohen_kappa = (observed_agreement - expected_agreement) / (
                1 - expected_agreement
            )

    return {
        "criteria": total,
        "graded_criteria": count,
        "failed_criteria": total - count,
        "coverage": _ratio(count, total),
        "strict_accuracy": _ratio(correct, total),
        "conditional_accuracy": observed_agreement,
        "ground_truth_positive_rate": _ratio(
            sum(row["ground_truth"] == 1 for row in rows), total
        ),
        "predicted_positive_rate": _ratio(tp + fp, count),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation": mcc,
        "cohen_kappa": cohen_kappa,
    }


def _percentile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(proportion * len(ordered)) - 1)
    return ordered[index]


def _usage_metrics(calls: list[CallResult]) -> dict[str, Any]:
    usage_records = [call.usage for call in calls if call.usage is not None]
    fields = (
        "prompt_tokens",
        "cached_input_tokens",
        "input_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cost",
    )
    totals: dict[str, float | int | None] = {}
    for field in fields:
        values = [
            usage[field] for usage in usage_records if usage.get(field) is not None
        ]
        totals[field] = sum(values) if values else None
    return {
        "calls_with_usage": len(usage_records),
        "totals": totals,
        "mean_total_tokens_per_call": _ratio(
            totals["total_tokens"] or 0,
            len(usage_records),
        ),
    }


def _call_metrics(calls: list[CallResult]) -> dict[str, Any]:
    durations = [call.elapsed_seconds for call in calls]
    successful = sum(call.error is None for call in calls)
    error_types = Counter(
        call.error.split(":", maxsplit=1)[0] for call in calls if call.error is not None
    )
    return {
        "calls": len(calls),
        "successful_calls": successful,
        "failed_calls": len(calls) - successful,
        "call_success_rate": _ratio(successful, len(calls)),
        "failure_types": dict(sorted(error_types.items())),
        "latency_seconds": {
            "mean": statistics.fmean(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "p95": _percentile(durations, 0.95),
            "max": max(durations) if durations else None,
        },
        "usage": _usage_metrics(calls),
    }


def _example_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_example[row["example_id"]].append(row)
    complete = [
        example_rows
        for example_rows in by_example.values()
        if all(row["prediction"] is not None for row in example_rows)
    ]
    exact = sum(
        all(row["prediction"] == row["ground_truth"] for row in example_rows)
        for example_rows in by_example.values()
    )
    complete_exact = sum(
        all(row["prediction"] == row["ground_truth"] for row in example_rows)
        for example_rows in complete
    )
    return {
        "examples": len(by_example),
        "complete_examples": len(complete),
        "complete_example_rate": _ratio(len(complete), len(by_example)),
        "strict_exact_match_rate": _ratio(exact, len(by_example)),
        "exact_match_rate_on_complete_examples": _ratio(
            complete_exact,
            len(complete),
        ),
    }


def _experiment_metrics(
    requested_size: int,
    rows: list[dict[str, Any]],
    calls: list[CallResult],
) -> dict[str, Any]:
    by_actual_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    calls_by_actual_size: dict[int, list[CallResult]] = defaultdict(list)
    by_position: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_actual_size[row["actual_criteria_in_call"]].append(row)
        by_position[row["position_in_call"]].append(row)
    for call in calls:
        calls_by_actual_size[call.actual_size].append(call)

    actual_size_metrics = {
        str(actual_size): {
            "criteria_metrics": _classification_metrics(actual_rows),
            "call_metrics": _call_metrics(calls_by_actual_size[actual_size]),
        }
        for actual_size, actual_rows in sorted(by_actual_size.items())
    }
    position_metrics = {
        str(position): _classification_metrics(position_rows)
        for position, position_rows in sorted(by_position.items())
    }
    return {
        "criteria_per_call": requested_size,
        "criteria_metrics": _classification_metrics(rows),
        "example_metrics": _example_metrics(rows),
        "call_metrics": _call_metrics(calls),
        "by_actual_criteria_in_call": actual_size_metrics,
        "by_position_in_call": position_metrics,
    }


def _comparisons(
    criteria_per_call: list[int],
    rows: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    baseline_size = 1 if 1 in criteria_per_call else min(criteria_per_call)
    by_size: dict[int, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["example_id"], row["criterion_name"])
        by_size[row["criteria_per_call"]][key] = row
    experiment_by_size = {
        experiment["criteria_per_call"]: experiment for experiment in experiments
    }
    baseline = by_size[baseline_size]
    baseline_accuracy = experiment_by_size[baseline_size]["criteria_metrics"][
        "strict_accuracy"
    ]

    comparisons: list[dict[str, Any]] = []
    for size in criteria_per_call:
        if size == baseline_size:
            continue
        target = by_size[size]
        jointly_graded = []
        regressions = 0
        improvements = 0
        agreements = 0
        for key, baseline_row in baseline.items():
            target_row = target[key]
            baseline_prediction = baseline_row["prediction"]
            target_prediction = target_row["prediction"]
            if baseline_prediction is None or target_prediction is None:
                continue
            jointly_graded.append(key)
            agreements += baseline_prediction == target_prediction
            baseline_correct = baseline_prediction == baseline_row["ground_truth"]
            target_correct = target_prediction == target_row["ground_truth"]
            regressions += baseline_correct and not target_correct
            improvements += not baseline_correct and target_correct

        target_accuracy = experiment_by_size[size]["criteria_metrics"][
            "strict_accuracy"
        ]
        delta = (
            target_accuracy - baseline_accuracy
            if target_accuracy is not None and baseline_accuracy is not None
            else None
        )
        comparisons.append(
            {
                "baseline_criteria_per_call": baseline_size,
                "criteria_per_call": size,
                "strict_accuracy_delta": delta,
                "strict_accuracy_degradation": -delta if delta is not None else None,
                "jointly_graded_criteria": len(jointly_graded),
                "prediction_agreement": _ratio(agreements, len(jointly_graded)),
                "prediction_flip_rate": _ratio(
                    len(jointly_graded) - agreements,
                    len(jointly_graded),
                ),
                "regressions": regressions,
                "improvements": improvements,
                "net_regressions": regressions - improvements,
            }
        )
    return baseline_size, comparisons


def _build_metrics(
    examples: list[GroundTruthExample],
    rows: list[dict[str, Any]],
    calls: list[CallResult],
    *,
    criteria_per_call: list[int],
    ground_truth_path: Path,
    judge_model: str,
    base_url: str,
    api_key_var: str,
    temperature: float | None,
    max_tokens: int,
    max_concurrent: int,
) -> dict[str, Any]:
    experiments = []
    for size in criteria_per_call:
        size_rows = [row for row in rows if row["criteria_per_call"] == size]
        size_calls = [call for call in calls if call.requested_size == size]
        experiments.append(_experiment_metrics(size, size_rows, size_calls))
    baseline_size, comparisons = _comparisons(
        criteria_per_call,
        rows,
        experiments,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "ground_truth_path": str(ground_truth_path.expanduser().resolve()),
        "judge": {
            "model": judge_model,
            "base_url": base_url,
            "api_key_var": api_key_var,
            "sampling": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "max_concurrent": max_concurrent,
        },
        "dataset": {
            "examples": len(examples),
            "criteria": sum(len(example.criteria) for example in examples),
            "ground_truth_positive_rate": _ratio(
                sum(
                    criterion.score
                    for example in examples
                    for criterion in example.criteria
                ),
                sum(len(example.criteria) for example in examples),
            ),
        },
        "baseline_criteria_per_call": baseline_size,
        "experiments": experiments,
        "comparisons_to_baseline": comparisons,
    }


def _write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _format_percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _format_cost(value: float | int | None) -> str:
    return "-" if value is None else f"${float(value):.4f}"


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.rjust(widths[index]) for index, value in enumerate(row))

    return [
        render(headers),
        render(tuple("-" * width for width in widths)),
        *map(render, rows),
    ]


def _print_summary(metrics: dict[str, Any]) -> None:
    dataset = metrics["dataset"]
    judge = metrics["judge"]
    lines = [
        "Rubric judge evaluation summary",
        f"Judge: {judge['model']}",
        (
            f"Dataset: {dataset['examples']} examples, {dataset['criteria']} criteria, "
            f"{_format_percent(dataset['ground_truth_positive_rate'])} positive"
        ),
        "",
    ]

    experiment_rows: list[tuple[str, ...]] = []
    for experiment in metrics["experiments"]:
        criteria = experiment["criteria_metrics"]
        examples = experiment["example_metrics"]
        calls = experiment["call_metrics"]
        usage = calls["usage"]["totals"]
        experiment_rows.append(
            (
                str(experiment["criteria_per_call"]),
                f"{calls['successful_calls']}/{calls['calls']}",
                _format_percent(criteria["coverage"]),
                _format_percent(criteria["strict_accuracy"]),
                _format_percent(criteria["conditional_accuracy"]),
                f"{criteria['false_positives']}/{criteria['false_negatives']}",
                _format_percent(examples["strict_exact_match_rate"]),
                f"{calls['latency_seconds']['mean']:.1f}",
                f"{usage['total_tokens']:,}"
                if usage["total_tokens"] is not None
                else "-",
                _format_cost(usage["cost"]),
            )
        )
    lines.extend(
        _format_table(
            (
                "Criteria/call",
                "Calls OK",
                "Coverage",
                "Strict acc",
                "Cond. acc",
                "FP/FN",
                "Exact",
                "Mean sec",
                "Tokens",
                "Cost",
            ),
            experiment_rows,
        )
    )

    comparisons = metrics["comparisons_to_baseline"]
    if comparisons:
        lines.extend(
            [
                "",
                (
                    "Comparison with baseline "
                    f"({metrics['baseline_criteria_per_call']} criterion/call)"
                ),
            ]
        )
        comparison_rows = [
            (
                str(comparison["criteria_per_call"]),
                _format_percent(comparison["strict_accuracy_delta"], signed=True),
                _format_percent(comparison["prediction_agreement"]),
                _format_percent(comparison["prediction_flip_rate"]),
                str(comparison["regressions"]),
                str(comparison["improvements"]),
            )
            for comparison in comparisons
        ]
        lines.extend(
            _format_table(
                (
                    "Criteria/call",
                    "Strict acc delta",
                    "Agreement",
                    "Flip rate",
                    "Regressions",
                    "Improvements",
                ),
                comparison_rows,
            )
        )

    print("\n".join(lines), flush=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    criteria_per_call = list(dict.fromkeys(args.criteria_per_call))
    ground_truth_path = _resolve_traces_path(args.ground_truth_path)
    metrics_output = (
        args.metrics_output.expanduser().resolve()
        if args.metrics_output
        else ground_truth_path.with_name(f"{ground_truth_path.stem}.metrics.json")
    )
    judgments_output = (
        args.judgments_output.expanduser().resolve()
        if args.judgments_output
        else metrics_output.with_name(f"{metrics_output.stem}.judgments.jsonl")
    )
    if metrics_output == judgments_output:
        raise ValueError(
            "--metrics-output and --judgments-output must be different files"
        )
    for path in (metrics_output, judgments_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output file already exists: {path}; pass --overwrite"
            )

    examples = _load_ground_truth(ground_truth_path)
    rows, calls = asyncio.run(
        _run_experiments(
            examples,
            criteria_per_call,
            judge_model=args.judge_model,
            base_url=args.base_url,
            api_key_var=args.api_key_var,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
        )
    )
    metrics = _build_metrics(
        examples,
        rows,
        calls,
        criteria_per_call=criteria_per_call,
        ground_truth_path=ground_truth_path,
        judge_model=args.judge_model,
        base_url=args.base_url,
        api_key_var=args.api_key_var,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
    )
    _write_jsonl(judgments_output, rows, args.overwrite)
    _write_json(metrics_output, metrics, args.overwrite)

    _print_summary(metrics)
    logger.info("Wrote detailed metrics to %s", metrics_output)
    logger.info("Wrote criterion-level judgments to %s", judgments_output)


if __name__ == "__main__":
    main()
