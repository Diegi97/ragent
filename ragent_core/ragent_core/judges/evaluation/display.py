from typing import Any


def print_summary(metrics: dict[str, Any]) -> None:
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
