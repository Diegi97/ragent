import hashlib
from pathlib import Path
from typing import Any

from data_pipelines.artifacts.inputs import read_json
from data_pipelines.artifacts.io import write_json
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AuditExecutionError,
    AuditPaths,
    AuditState,
    RetrievalAudit,
    SolverAudit,
    ValidatedAudits,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.difficulty import (
    MAX_REJECTED_SOLVER_PASS_PERCENT,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from ragent_core.artifacts.question_rubric import (
    QuestionRubricRecord,
)
from ragent_core.judges.criteria import criterion_id


def begin_candidate_audit(
    candidate: Path, audit_path: Path
) -> tuple[QuestionRubricRecord, str]:
    digest = question_rubric_sha256(candidate)
    # Invalidate prior success before validation or any remote execution can fail.
    write_json(
        audit_path,
        AuditState(ok=False, candidate_sha256=digest).model_dump(mode="json"),
    )
    return validate_question_rubric_file(candidate), digest


def question_rubric_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_audit(path: Path, *, label: str, digest: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label} audit for final candidate: {path}")
    payload = read_json(path)
    state = AuditState.model_validate(payload)
    if state.candidate_sha256 == digest and not state.ok:
        raise AuditExecutionError(
            f"{label} execution did not complete for the final candidate"
        )
    return payload


def validate_question_rubric_audits(
    path: Path,
    audits_directory: Path,
    record: QuestionRubricRecord,
) -> ValidatedAudits:
    digest = question_rubric_sha256(path)
    paths = AuditPaths(audits_directory, path.name)
    retrieval = RetrievalAudit.model_validate(
        _load_audit(paths.retrieval, label="retrieval probe", digest=digest)
    )
    solver = SolverAudit.model_validate(
        _load_audit(paths.solver, label="solver", digest=digest)
    )
    for label, audit in (("retrieval probe", retrieval), ("solver", solver)):
        if audit.candidate_sha256 != digest:
            raise ValueError(f"{label} audit does not match the final candidate")
        if audit.ok is not True:
            raise ValueError(f"{label} audit did not complete successfully")
        if audit.question != record.question:
            raise ValueError(
                f"{label} audit question does not match the final candidate"
            )

    retrieval_doc_ids = retrieval.supporting_doc_ids
    if retrieval_doc_ids != record.doc_ids:
        raise ValueError(
            "retrieval probe supporting_doc_ids do not match final candidate Docs"
        )
    if retrieval.probe_passed is not True:
        raise ValueError(
            "final candidate did not pass the retrieval gate; evolve it before solving"
        )
    if solver.criteria_total != len(record.rubric):
        raise ValueError("solver audit criterion count does not match final rubric")
    expected_criteria = {
        criterion_id(index) for index in range(1, len(record.rubric) + 1)
    }
    if {judgment.id for judgment in solver.judgments} != expected_criteria:
        raise ValueError("solver audit judgments do not cover the final rubric")
    percent_passed = solver.percent_passed
    if percent_passed <= MAX_REJECTED_SOLVER_PASS_PERCENT:
        raise ValueError(
            f"solver passed {MAX_REJECTED_SOLVER_PASS_PERCENT}% or fewer criteria; "
            "inspect and repair or discard the item"
        )

    return ValidatedAudits(retrieval=retrieval, solver=solver)
