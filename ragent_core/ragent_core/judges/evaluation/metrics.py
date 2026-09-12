import logging
import math
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragent_core.judges.evaluation.models import (
    SCHEMA_VERSION,
    CallResult,
    GroundTruthExample,
)

logger = logging.getLogger(__name__)


def build_metrics(
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
        key = (row["example_id"], row["criterion_id"])
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
