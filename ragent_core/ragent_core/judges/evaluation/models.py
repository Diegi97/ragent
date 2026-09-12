from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ragent_core.judges.criteria import Verdict, criterion_metric_name
from ragent_core.judges.criteria import criterion_id as make_criterion_id

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class GroundTruthCriterion:
    id: str
    text: str
    score: int
    index: int


@dataclass(frozen=True)
class GroundTruthExample:
    example_id: str
    question: str
    response: str
    criteria: tuple[GroundTruthCriterion, ...]

    @classmethod
    def from_trace(cls, trace: dict[str, Any]) -> GroundTruthExample:
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
                raw_criterion.get("criterion")
                if isinstance(raw_criterion, dict)
                else None
            )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"rubric criterion {criterion_index} has no text")
            criterion_items.append(
                (make_criterion_id(criterion_index), text.strip(), criterion_index)
            )

        expected_ids = {criterion_id for criterion_id, _, _ in criterion_items}
        legacy_ids = {
            f"criterion_{criterion_index:02d}": criterion_id
            for criterion_id, _, criterion_index in criterion_items
        }
        teacher_scores = _teacher_scores(trace, expected_ids, legacy_ids)
        metrics = trace.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("trace contains no criterion metrics")

        criteria: list[GroundTruthCriterion] = []
        for criterion_id, text, criterion_index in criterion_items:
            metric_name = criterion_metric_name(criterion_id)
            metric = metrics.get(metric_name)
            if metric is None:
                metric = metrics.get(f"rubric/criterion_{criterion_index:02d}")
            if metric not in (0, 0.0, 1, 1.0):
                raise ValueError(f"trace contains no binary {metric_name!r} metric")
            score = teacher_scores[criterion_id]
            if int(metric) != score:
                raise ValueError(f"teacher verdict and {metric_name!r} disagree")
            criteria.append(
                GroundTruthCriterion(
                    id=criterion_id,
                    text=text,
                    score=score,
                    index=criterion_index,
                )
            )

        return cls(
            example_id=example_id,
            question=question.strip(),
            response=_trace_response(trace),
            criteria=tuple(criteria),
        )


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
    expected_ids: set[str],
    legacy_ids: dict[str, str],
) -> dict[str, int]:
    info = trace.get("info")
    calls = info.get("judge") if isinstance(info, dict) else None
    if not isinstance(calls, list) or len(calls) != len(expected_ids):
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
        criterion_id = verdict.get("id", verdict.get("name"))
        teacher_verdict = verdict.get("verdict")
        if isinstance(criterion_id, str):
            criterion_id = legacy_ids.get(criterion_id, criterion_id)
        if not isinstance(criterion_id, str) or criterion_id not in expected_ids:
            raise ValueError(
                "teacher call "
                f"{call_index} returned unexpected criterion {criterion_id!r}"
            )
        if criterion_id in scores:
            raise ValueError(f"teacher returned duplicate verdict for {criterion_id!r}")
        if not isinstance(teacher_verdict, str):
            raise ValueError(f"teacher call {call_index} has no verdict")
        try:
            scores[criterion_id] = int(Verdict.from_teacher(teacher_verdict).score)
        except ValueError as exc:
            raise ValueError(
                f"teacher returned invalid verdict {teacher_verdict!r} for {criterion_id}"
            ) from exc

    if set(scores) != expected_ids:
        raise ValueError("teacher judgments do not cover every rubric criterion")
    return scores
