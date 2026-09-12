import asyncio
import json

import pytest

from ragent_core.judges.criteria import Verdict, criterion_id, criterion_metric_name
from ragent_core.judges.evaluation import cli, experiments
from ragent_core.judges.evaluation.models import GroundTruthExample


def teacher_trace():
    identifiers = [criterion_id(index) for index in range(1, 4)]
    return {
        "id": "example",
        "task": {
            "data": {
                "question": "Question",
                "rubric": [
                    {"criterion": f"Requirement {identifier}"}
                    for identifier in identifiers
                ],
            }
        },
        "nodes": [
            {"sampled": True, "message": {"role": "assistant", "content": "Answer"}}
        ],
        "info": {
            "judge": [
                {"parsed": [{"id": identifier, "verdict": Verdict.PASS}]}
                for identifier in identifiers
            ]
        },
        "metrics": {criterion_metric_name(identifier): 1 for identifier in identifiers},
    }


@pytest.fixture
def fake_experiment_judge(monkeypatch):
    calls = []

    class Judge:
        def __init__(self, config):
            self.config = config

        async def grade_batch(self, *, trace, question, response, batch):
            calls.append((self.config.max_criteria, [item.id for item in batch]))
            await asyncio.sleep(0)
            if len(batch) == 2:
                raise RuntimeError("judge unavailable")
            trace.info["judge"] = [
                {
                    "text": "fake response",
                    "parsed": [
                        {"id": item.id, "reason": "supported", "verdict": Verdict.PASS}
                        for item in batch
                    ],
                }
            ]
            return {item.id: 1.0 for item in batch}

    monkeypatch.setattr(experiments, "RubricJudge", Judge)
    return calls


def test_experiment_batching_order_and_failure_isolation(fake_experiment_judge):
    example = GroundTruthExample.from_trace(teacher_trace())
    rows, calls = asyncio.run(
        experiments.run_experiments(
            [example],
            [1, 2],
            judge_model="fake",
            base_url="http://judge.test",
            api_key_var="TEST_KEY",
            temperature=0,
            max_tokens=100,
            max_concurrent=2,
        )
    )
    assert [call.actual_size for call in calls] == [1, 1, 1, 2, 1]
    assert [row["criterion_id"] for row in rows] == [
        criterion_id(index) for index in [1, 2, 3, 1, 2, 3]
    ]
    assert [row["prediction"] for row in rows] == [1, 1, 1, None, None, 1]
    assert calls[3].error == "RuntimeError: judge unavailable"
    assert rows[0]["judge_response"] == "fake response"
    assert len(fake_experiment_judge) == 5


def test_judge_cli_persists_experiments_and_refuses_unapproved_overwrite(
    tmp_path, monkeypatch, fake_experiment_judge
):
    source = tmp_path / "traces.jsonl"
    source.write_text(json.dumps(teacher_trace()) + "\n")
    metrics = tmp_path / "metrics.json"
    judgments = tmp_path / "judgments.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_rubric_judge",
            str(source),
            "fake",
            "--criteria-per-call",
            "1",
            "2",
            "1",
            "--metrics-output",
            str(metrics),
            "--judgments-output",
            str(judgments),
        ],
    )
    cli.main()
    report = json.loads(metrics.read_text())
    rows = [json.loads(line) for line in judgments.read_text().splitlines()]
    assert report["dataset"]["criteria"] == 3
    assert len(report["experiments"]) == 2
    assert len(rows) == 6
    assert sum(row["prediction"] is None for row in rows) == 2
    with pytest.raises(FileExistsError):
        cli.main()
    assert len(fake_experiment_judge) == 5
