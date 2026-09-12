from typing import Any

from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AuditVerdict,
    CriterionJudgment,
    SolverAudit,
    criteria_pass_percentage,
)
from ragent_core.artifacts.question_rubric import QuestionRubricRecord
from ragent_core.judges.criteria import criterion_id, criterion_metric_name
from ragent_deep_search.citations import citation_ids


def _judge_verdicts(trace: Any) -> dict[str, AuditVerdict]:
    verdicts: dict[str, AuditVerdict] = {}
    for judge_record in trace.info.get("judge", []):
        for verdict in judge_record.get("parsed") or []:
            criterion_key = str(verdict.get("id") or "")
            if criterion_key:
                verdicts[criterion_key] = AuditVerdict.model_validate(verdict)
    return verdicts


def error_detail(value: Any) -> str:
    last_error = getattr(value, "last_error", None)
    if last_error is not None:
        return str(getattr(last_error, "message", last_error))
    errors = getattr(value, "errors", None) or []
    if errors:
        error = errors[-1]
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(getattr(error, "message", error))
    return "unknown error"


def solver_report(
    record: QuestionRubricRecord, trace: Any, digest: str
) -> dict[str, Any]:
    if not trace.ok:
        raise RuntimeError(f"solver rollout failed: {error_detail(trace)}")
    verdicts = _judge_verdicts(trace)
    judgments: list[CriterionJudgment] = []
    passed = 0
    for index, criterion in enumerate(record.rubric, start=1):
        criterion_key = criterion_id(index)
        metric = trace.metrics.get(criterion_metric_name(criterion_key))
        if metric is None:
            raise RuntimeError(
                f"solver trace is missing metric {criterion_metric_name(criterion_key)}"
            )
        passed_criterion = metric == 1.0
        passed += int(passed_criterion)
        judgment = verdicts.get(criterion_key, AuditVerdict())
        judgments.append(
            CriterionJudgment(
                id=criterion_key,
                criterion=criterion.criterion,
                doc_ids=criterion.doc_ids,
                passed=passed_criterion,
                verdict=judgment.verdict,
                reason=judgment.reason,
            )
        )
    total = len(judgments)
    answer = trace.last_reply
    return SolverAudit.model_validate(
        {
            "ok": True,
            "candidate_sha256": digest,
            "question": record.question,
            "answer": answer,
            "cited_doc_ids": list(dict.fromkeys(citation_ids(answer))),
            "judgments": judgments,
            "criteria_passed": passed,
            "criteria_total": total,
            "percent_passed": criteria_pass_percentage(passed, total),
        }
    ).model_dump(mode="json")
