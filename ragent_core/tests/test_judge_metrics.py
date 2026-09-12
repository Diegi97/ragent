from pathlib import Path

import pytest

from ragent_core.judges.evaluation.metrics import build_metrics


def metrics(rows):
    return build_metrics(
        [],
        rows,
        [],
        criteria_per_call=[1, 2],
        ground_truth_path=Path("fixture.jsonl"),
        judge_model="fake",
        base_url="http://localhost",
        api_key_var="TEST_KEY",
        temperature=0,
        max_tokens=100,
        max_concurrent=1,
    )


def test_judge_metrics_compare_predictions_and_report_failed_coverage():
    rows = [
        {
            "example_id": "example",
            "criterion_id": str(index),
            "criteria_per_call": size,
            "actual_criteria_in_call": size,
            "position_in_call": 1,
            "ground_truth": expected,
            "prediction": predicted,
        }
        for size, predictions in [(1, [1, 0, 1, None]), (2, [0, 0, 1, 1])]
        for index, (expected, predicted) in enumerate(zip([1, 0, 0, 1], predictions))
    ]
    result = metrics(rows)
    baseline = result["experiments"][0]["criteria_metrics"]
    assert baseline["coverage"] == 0.75
    assert baseline["strict_accuracy"] == 0.5
    assert baseline["conditional_accuracy"] == pytest.approx(2 / 3)
    comparison = result["comparisons_to_baseline"][0]
    assert comparison["prediction_agreement"] == pytest.approx(2 / 3)
    assert comparison["regressions"] == 1
    assert comparison["improvements"] == 0


def test_empty_judge_metrics_keep_undefined_values_null():
    result = metrics([])
    scores = result["experiments"][0]["criteria_metrics"]
    assert scores["criteria"] == 0
    assert scores["coverage"] is None
    assert scores["f1"] is None
    assert scores["matthews_correlation"] is None
    assert result["comparisons_to_baseline"][0]["prediction_agreement"] is None
