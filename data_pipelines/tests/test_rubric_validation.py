import json
import subprocess
import sys

import pytest

from data_pipelines.pipelines.deep_search_task_generation.facts import (
    EntityFactMemoryRecord,
    ExtractedFact,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.audit_contract import (
    AuditExecutionError,
    AuditPaths,
    AuditState,
    AuditVerdict,
    RetrievalAudit,
    SolverAudit,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.difficulty import (
    MAX_REJECTED_SOLVER_PASS_PERCENT,
    DifficultyBandName,
    difficulty_band,
    difficulty_thresholds,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.profile import (
    build_dataset_profile,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation import (
    validate_question_rubric_file,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.audits import (
    begin_candidate_audit,
    question_rubric_sha256,
    validate_question_rubric_audits,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.validation.markdown import (
    example_markdown,
)
from data_pipelines.pipelines.deep_search_task_generation.generate.rubrics.workspace import (
    create_fact_workspace,
)
from ragent_core.judges.criteria import Verdict, criterion_id


def candidate_and_audits(tmp_path):
    candidate = tmp_path / "question_rubric_000000.md"
    candidate.write_text(example_markdown())
    record = validate_question_rubric_file(candidate)
    digest = question_rubric_sha256(candidate)
    retrieval = {
        "ok": True,
        "candidate_sha256": digest,
        "question": record.question,
        "supporting_doc_ids": record.doc_ids,
        "probe_passed": True,
    }
    solver = {
        "ok": True,
        "candidate_sha256": digest,
        "question": record.question,
        "criteria_total": 2,
        "criteria_passed": 1,
        "percent_passed": 50,
        "judgments": [
            {"id": criterion_id(1), "passed": True},
            {"id": criterion_id(2), "passed": False},
        ],
    }
    paths = AuditPaths(tmp_path, candidate.name)
    paths.retrieval.write_text(json.dumps(retrieval))
    paths.solver.write_text(json.dumps(solver))
    return candidate, record, paths, solver


def test_markdown_and_audits_feed_profile_without_rereading(tmp_path):
    candidate, record, paths, _ = candidate_and_audits(tmp_path)
    audits = validate_question_rubric_audits(candidate, tmp_path, record)
    paths.solver.unlink()
    profile = build_dataset_profile({0: record}, {0: audits.solver})
    assert profile["difficulty"]["distribution"][DifficultyBandName.HARD] == 1
    assert profile["solver_audits"]["criteria_passed"] == 1
    assert profile["solver_audits"]["missing_criterion_judgment_count"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_sha256", "changed"),
        ("question", "other question"),
        ("criteria_total", 3),
        ("percent_passed", MAX_REJECTED_SOLVER_PASS_PERCENT),
        ("percent_passed", float("nan")),
        ("percent_passed", True),
        ("percent_passed", 101),
    ],
)
def test_invalid_solver_audit_cannot_accept_candidate(tmp_path, field, value):
    candidate, record, paths, solver = candidate_and_audits(tmp_path)
    solver[field] = value
    paths.solver.write_text(json.dumps(solver))
    with pytest.raises(ValueError):
        validate_question_rubric_audits(candidate, tmp_path, record)


def test_failed_probe_blocks_acceptance(tmp_path):
    candidate, record, paths, _ = candidate_and_audits(tmp_path)
    retrieval = json.loads(paths.retrieval.read_text())
    retrieval["probe_passed"] = False
    paths.retrieval.write_text(json.dumps(retrieval))
    with pytest.raises(ValueError, match="retrieval gate"):
        validate_question_rubric_audits(candidate, tmp_path, record)


def test_missing_audit_is_explicit_in_profile(tmp_path):
    _, record, _, _ = candidate_and_audits(tmp_path)
    profile = build_dataset_profile({0: record}, {})
    assert profile["difficulty"]["distribution"][DifficultyBandName.UNKNOWN] == 1
    assert profile["solver_audits"]["audit_error_count"] == 1


def test_difficulty_boundaries_share_display_table():
    assert [difficulty_band(x) for x in [0, 39.9, 40, 59.9, 60, 84.9, 85, 100]] == [
        DifficultyBandName.VERY_HARD,
        DifficultyBandName.VERY_HARD,
        DifficultyBandName.HARD,
        DifficultyBandName.HARD,
        DifficultyBandName.MIDDLE,
        DifficultyBandName.MIDDLE,
        DifficultyBandName.EASY,
        DifficultyBandName.EASY,
    ]
    assert difficulty_thresholds()[DifficultyBandName.MIDDLE] == "60-<85"


def test_workspace_launchers_run_from_outside_repository(tmp_path):
    workspace = create_fact_workspace(
        tmp_path / "workspace",
        [
            EntityFactMemoryRecord(
                entity_name="Entity",
                data_source="source",
                entity_doc_ids=(123, 456, 789),
                facts=(ExtractedFact("Fact", [123, 456, 789]),),
            )
        ],
    )
    candidate = workspace.outputs_directory / "candidate.md"
    candidate.write_text(example_markdown())
    result = subprocess.run(
        [sys.executable, str(workspace.validator), str(candidate)],
        cwd=workspace.directory,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    for script in [workspace.retrieval_probe, workspace.solver]:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=workspace.directory,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert not script.stat().st_mode & 0o222


def test_failed_audit_execution_is_distinct_from_semantic_rejection(tmp_path):

    candidate, record, paths, _ = candidate_and_audits(tmp_path)
    prior = json.loads(paths.solver.read_text())
    failed = AuditState(ok=False, candidate_sha256=prior["candidate_sha256"])
    paths.solver.write_text(failed.model_dump_json())
    with pytest.raises(AuditExecutionError, match="execution did not complete"):
        validate_question_rubric_audits(candidate, tmp_path, record)


def test_begin_audit_invalidates_prior_success_before_candidate_validation(tmp_path):
    candidate = tmp_path / "candidate.md"
    candidate.write_text("invalid rubric")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"ok": True, "candidate_sha256": "stale"}))
    with pytest.raises(ValueError):
        begin_candidate_audit(candidate, audit_path)
    failed = json.loads(audit_path.read_text())
    assert failed["ok"] is False
    assert failed["candidate_sha256"] == question_rubric_sha256(candidate)


@pytest.mark.parametrize(
    "updates",
    [
        {"criteria_passed": 3},
        {"criteria_passed": 0, "percent_passed": 100, "judgments": []},
        {"judgments": []},
        {
            "judgments": [
                {"id": criterion_id(1), "passed": True},
                {"id": criterion_id(1), "passed": False},
            ]
        },
        {
            "judgments": [
                {"id": criterion_id(1), "passed": False},
                {"id": criterion_id(2), "passed": False},
            ]
        },
    ],
)
def test_inconsistent_audit_results_cannot_accept_candidate(tmp_path, updates):
    candidate, record, paths, solver = candidate_and_audits(tmp_path)
    paths.solver.write_text(json.dumps(solver | updates))
    with pytest.raises(ValueError):
        validate_question_rubric_audits(candidate, tmp_path, record)


def test_profile_keeps_missing_judgment_diagnostics_for_historical_audits(tmp_path):
    _, record, _, solver = candidate_and_audits(tmp_path)
    audit = SolverAudit.model_validate(solver | {"judgments": []})
    profile = build_dataset_profile({0: record}, {0: audit})
    assert profile["solver_audits"]["missing_criterion_judgment_count"] == 2


def test_retrieval_audit_preserves_legacy_json_key():
    audit = RetrievalAudit(
        ok=True,
        candidate_sha256="hash",
        question="Question",
        supporting_doc_ids=[0],
        all_supporting_docs_retrieved=True,
        probe_passed=False,
    )
    payload = audit.model_dump(mode="json")
    assert payload["all_supporting_docs_in_top_10"] is True
    assert "all_supporting_docs_retrieved" not in payload
    assert RetrievalAudit.model_validate(payload).all_supporting_docs_retrieved is True


def test_consistent_solver_failure_is_rejected_by_integrity_threshold(tmp_path):
    candidate, record, paths, solver = candidate_and_audits(tmp_path)
    solver.update(criteria_passed=0, percent_passed=0)
    for judgment in solver["judgments"]:
        judgment["passed"] = False
    paths.solver.write_text(json.dumps(solver))
    with pytest.raises(
        ValueError, match=f"{MAX_REJECTED_SOLVER_PASS_PERCENT}% or fewer"
    ):
        validate_question_rubric_audits(candidate, tmp_path, record)


@pytest.mark.parametrize(
    "recorded,expected", [("yes", Verdict.PASS), ("no", Verdict.FAIL), ("", None)]
)
def test_audit_verdict_preserves_historical_absence_and_normalizes_aliases(
    recorded, expected
):
    verdict = AuditVerdict.model_validate({"verdict": recorded, "reason": "detail"})
    assert verdict.verdict is expected
    payload = verdict.model_dump(mode="json")
    assert payload["verdict"] == (expected.value if expected is not None else "")
    assert AuditVerdict.model_validate(payload) == verdict
    with pytest.raises(ValueError):
        AuditVerdict.model_validate({"verdict": "MAYBE"})
