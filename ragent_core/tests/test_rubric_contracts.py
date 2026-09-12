import asyncio
from types import SimpleNamespace

import pytest

from ragent_core.judges.criteria import Verdict, criterion_id, criterion_metric_name
from ragent_core.judges.evaluation.models import GroundTruthExample
from ragent_core.judges.rubric import RubricJudge
from ragent_core.judges.rubric.config import RubricJudgeConfig
from ragent_core.judges.rubric.contracts import CriterionVerdict, JudgeCriterion


def test_verdict_parser_uses_last_valid_block_and_unescaped_reason():
    first = f"<criteria><criterion><id>{criterion_id(1)}</id><reason>old</reason><verdict>FAIL</verdict></criterion></criteria>"
    last = f"<criteria><criterion><id>{criterion_id(1)}</id><reason>A & B</reason><verdict>pass</verdict></criterion></criteria>"
    parsed = CriterionVerdict.parse_xml(first + last)
    assert parsed[0].verdict is Verdict.PASS
    assert parsed[0].reason == "A & B"


def test_invalid_verdict_is_rejected_at_parser_boundary():
    with pytest.raises(ValueError, match="no parseable criteria"):
        CriterionVerdict.parse_xml(
            f"<criteria><criterion><id>{criterion_id(1)}</id><reason>x</reason><verdict>MAYBE</verdict></criterion></criteria>"
        )


@pytest.mark.parametrize(
    "teacher_verdict,score", [("PASS", 1), ("FAIL", 0), ("yes", 1), ("no", 0)]
)
def test_current_and_historical_teacher_traces(teacher_verdict, score):
    identifier = criterion_id(1)
    trace = {
        "id": "example",
        "task": {
            "data": {"question": "Question?", "rubric": [{"criterion": "Requirement"}]}
        },
        "nodes": [
            {"sampled": True, "message": {"role": "assistant", "content": "Answer"}}
        ],
        "info": {
            "judge": [{"parsed": [{"id": identifier, "verdict": teacher_verdict}]}]
        },
        "metrics": {criterion_metric_name(identifier): score},
    }
    example = GroundTruthExample.from_trace(trace)
    assert example.criteria[0].score == score
    trace["metrics"][criterion_metric_name(identifier)] = 1 - score
    with pytest.raises(ValueError, match="disagree"):
        GroundTruthExample.from_trace(trace)


def test_grade_batch_retries_duplicate_criterion_ids(monkeypatch):
    judge = RubricJudge(RubricJudgeConfig(model="test", max_retries=1))
    calls = []
    identifier = criterion_id(1)

    async def evaluate(**kwargs):
        calls.append(kwargs)
        verdict = CriterionVerdict(id=identifier, reason="reason", verdict=Verdict.PASS)
        return SimpleNamespace(
            parsed=[verdict, verdict] if len(calls) == 1 else [verdict]
        )

    monkeypatch.setattr(judge, "evaluate", evaluate)
    scores = asyncio.run(
        judge.grade_batch(
            trace=SimpleNamespace(),
            question="Question",
            response="Answer",
            batch=[JudgeCriterion(id=identifier, text="requirement")],
        )
    )
    assert scores == {identifier: Verdict.PASS.score}
    assert len(calls) == 2


def test_score_records_canonical_metrics_and_average(monkeypatch):
    judge = RubricJudge(RubricJudgeConfig(model="test", max_criteria=1))
    metrics = {}
    trace = SimpleNamespace(
        last_reply="Answer",
        record_metric=lambda key, value: metrics.__setitem__(key, value),
    )
    task = SimpleNamespace(
        question="Question", rubric=[{"criterion": "First"}, {"criterion": "Second"}]
    )

    async def grade_batch(**kwargs):
        identifier = kwargs["batch"][0].id
        return {identifier: float(identifier == criterion_id(1))}

    monkeypatch.setattr(judge, "grade_batch", grade_batch)
    assert asyncio.run(judge.score(task, trace)) == 0.5
    assert metrics == {
        criterion_metric_name(criterion_id(1)): 1.0,
        criterion_metric_name(criterion_id(2)): 0.0,
    }


def test_failed_judge_batch_cancels_other_batches(monkeypatch):
    judge = RubricJudge(RubricJudgeConfig(model="test", max_criteria=1))
    task = SimpleNamespace(
        question="Question", rubric=[{"criterion": "First"}, {"criterion": "Second"}]
    )
    trace = SimpleNamespace(last_reply="Answer")
    stopped = []

    async def run():
        both_started = asyncio.Event()
        started = []

        async def grade_batch(**kwargs):
            identifier = kwargs["batch"][0].id
            started.append(identifier)
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            if identifier == criterion_id(1):
                raise RuntimeError("judge unavailable")
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(identifier)

        monkeypatch.setattr(judge, "grade_batch", grade_batch)
        with pytest.raises(RuntimeError, match="judge unavailable"):
            await judge.score(task, trace)

    asyncio.run(run())
    assert stopped == [criterion_id(2)]
